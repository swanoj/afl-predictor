"""Player value model for AFL availability adjustments.

The market's biggest informational edge over a pure team-rating model is
*team news*: when rated players are OUT (injury/rest/suspension) the line
moves. This module estimates a per-player *value* so that the strength of a
named lineup can be summed and compared to a team's baseline lineup.

Approach (a) — the robust baseline implemented here — scores each player with
an AFL-fantasy-style points formula (weighted disposals, marks, tackles,
hit-outs, goals, behinds) and uses each player's **walk-forward** mean score
as their value. Values are shrunk toward a replacement level so that
small-sample / fringe players don't dominate.

The value is in "fantasy points". A team's lineup value is the sum over its
~22 players. The conversion from a lineup's fantasy-point surplus/deficit to a
*margin points* adjustment is handled by ``lineups.availability_adjustment``
via a fitted scalar (see ``DEFAULT_POINTS_PER_VALUE``).

Approach (b) — ridge regression of margin on per-player presence indicators —
is explored in ``scripts/eval_availability.py`` and compared honestly there.

Leakage discipline: ``compute_value_table(as_of_season)`` only ever reads
``PlayerGameLog`` rows with ``year < as_of_season``. The eval does an even
finer-grained chronological walk-forward (games strictly before each match).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import PlayerGameLog, PlayerValue

# ---------------------------------------------------------------------------
# AFL-fantasy-style scoring weights.
#
# Standard AFL Fantasy: kick 3, handball 2, mark 3, tackle 4, hit-out 1,
# goal 6, behind 1, free-for 1, free-against -3. We do not store frees, so
# they are omitted. We score kicks + handballs directly (rather than the
# `disposals` aggregate) and fall back to `disposals * 2.5` if the kick /
# handball split is missing for a row.
# ---------------------------------------------------------------------------
FANTASY_WEIGHTS: dict[str, float] = {
    "kicks": 3.0,
    "handballs": 2.0,
    "marks": 3.0,
    "tackles": 4.0,
    "hit_outs": 1.0,
    "goals": 6.0,
    "behinds": 1.0,
}

# Average value a player gets per disposal when only the disposal total is
# known (blend of the kick=3 / handball=2 weights, ~55% kicks historically).
_DISPOSAL_FALLBACK_WEIGHT = 2.5

# Shrinkage: a player's value is pulled toward the replacement level with a
# pseudo-count of this many "replacement games". Larger -> more conservative.
DEFAULT_SHRINKAGE = 8.0

# Minimum prior games for a player's raw mean to be considered meaningful;
# below this we still emit a (heavily shrunk) value.
DEFAULT_MIN_GAMES = 3

# Replacement level is the Nth percentile of established players' raw values.
# A debutant / unknown is assumed to be roughly a low-end senior-list player.
REPLACEMENT_PERCENTILE = 25.0

# Fitted fantasy-points -> margin-points scalar. Derived in
# scripts/eval_availability.py by regressing match margin on the home-minus-away
# lineup-value surplus while controlling for Elo (the Elo-controlled coefficient
# is the right one because this adjustment is added ALONGSIDE the team rating).
# Walk-forward fit on 2018-2025 realized lineups gives ~0.024 pts per fantasy
# point. NOTE: realized-lineup optimism means the true pre-game value is likely
# a bit smaller; treat this as an upper-ish bound.
DEFAULT_POINTS_PER_VALUE = 0.024


@dataclass(frozen=True)
class ValueRecord:
    player_name: str
    team: str
    value: float       # shrunk, walk-forward (fantasy points / game)
    raw_value: float   # unshrunk per-game mean
    games_sample: int


def fantasy_points(stats) -> float:
    """AFL-fantasy-style score for a single game log (or any object/dict with
    the stat attributes). Robust to missing kick/handball splits."""

    def g(name: str) -> float:
        if isinstance(stats, dict):
            return float(stats.get(name, 0) or 0)
        return float(getattr(stats, name, 0) or 0)

    kicks = g("kicks")
    handballs = g("handballs")
    disposals = g("disposals")

    if kicks <= 0 and handballs <= 0 and disposals > 0:
        # Only the aggregate is available; approximate the disposal value.
        score = disposals * _DISPOSAL_FALLBACK_WEIGHT
    else:
        score = kicks * FANTASY_WEIGHTS["kicks"] + handballs * FANTASY_WEIGHTS["handballs"]

    score += g("marks") * FANTASY_WEIGHTS["marks"]
    score += g("tackles") * FANTASY_WEIGHTS["tackles"]
    score += g("hit_outs") * FANTASY_WEIGHTS["hit_outs"]
    score += g("goals") * FANTASY_WEIGHTS["goals"]
    score += g("behinds") * FANTASY_WEIGHTS["behinds"]
    return score


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def shrink(raw_mean: float, games: int, replacement: float, shrinkage: float) -> float:
    """Empirical-Bayes style shrink of a player's mean toward replacement."""
    if games <= 0:
        return replacement
    return (games * raw_mean + shrinkage * replacement) / (games + shrinkage)


def compute_value_table(
    session: Session,
    as_of_season: int,
    *,
    min_games: int = DEFAULT_MIN_GAMES,
    shrinkage: float = DEFAULT_SHRINKAGE,
    persist: bool = True,
) -> dict[tuple[str, str], ValueRecord]:
    """Compute walk-forward player values from games strictly BEFORE
    ``as_of_season`` and (optionally) persist them to ``PlayerValue``.

    Returns a mapping ``(player_name, team) -> ValueRecord``.
    """
    logs = session.scalars(
        select(PlayerGameLog).where(PlayerGameLog.year < as_of_season)
    ).all()

    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for log in logs:
        key = (log.player_name, log.team)
        sums[key] = sums.get(key, 0.0) + fantasy_points(log)
        counts[key] = counts.get(key, 0) + 1

    # Replacement level = a low percentile of *established* players' means.
    established = [
        sums[k] / counts[k] for k in sums if counts[k] >= max(min_games, 5)
    ]
    replacement = _percentile(established, REPLACEMENT_PERCENTILE) if established else 0.0

    records: dict[tuple[str, str], ValueRecord] = {}
    for key, total in sums.items():
        n = counts[key]
        raw = total / n
        val = shrink(raw, n, replacement, shrinkage)
        records[key] = ValueRecord(
            player_name=key[0],
            team=key[1],
            value=val,
            raw_value=raw,
            games_sample=n,
        )

    if persist:
        _persist(session, as_of_season, records)

    return records


def _persist(
    session: Session,
    as_of_season: int,
    records: dict[tuple[str, str], ValueRecord],
) -> None:
    existing = {
        (r.player_name, r.team): r
        for r in session.scalars(
            select(PlayerValue).where(PlayerValue.as_of_season == as_of_season)
        ).all()
    }
    for key, rec in records.items():
        row = existing.get(key)
        if row is None:
            session.add(
                PlayerValue(
                    player_name=rec.player_name,
                    team=rec.team,
                    as_of_season=as_of_season,
                    value=rec.value,
                    raw_value=rec.raw_value,
                    games_sample=rec.games_sample,
                )
            )
        else:
            row.value = rec.value
            row.raw_value = rec.raw_value
            row.games_sample = rec.games_sample
    session.flush()


def replacement_value(values: dict[tuple[str, str], ValueRecord]) -> float:
    """Replacement level implied by a value table: the low-percentile value of
    players with a non-trivial sample. Used for unknown / debutant players."""
    pool = [r.value for r in values.values() if r.games_sample >= 5]
    if not pool:
        pool = [r.value for r in values.values()]
    return _percentile(pool, REPLACEMENT_PERCENTILE) if pool else 0.0


def value_for(
    values: dict[tuple[str, str], ValueRecord],
    player_name: str,
    team: str,
    replacement: float,
) -> float:
    """Look up a player's value, defaulting unknown players to replacement."""
    rec = values.get((player_name, team))
    return rec.value if rec is not None else replacement


def load_player_values(
    session: Session,
    as_of_season: int,
    *,
    compute_if_missing: bool = True,
) -> dict[tuple[str, str], ValueRecord]:
    """Load persisted ``PlayerValue`` rows for ``as_of_season``; compute and
    persist them on a cache miss."""
    rows = session.scalars(
        select(PlayerValue).where(PlayerValue.as_of_season == as_of_season)
    ).all()
    if rows:
        return {
            (r.player_name, r.team): ValueRecord(
                player_name=r.player_name,
                team=r.team,
                value=r.value,
                raw_value=r.raw_value,
                games_sample=r.games_sample,
            )
            for r in rows
        }
    if compute_if_missing:
        return compute_value_table(session, as_of_season, persist=True)
    return {}
