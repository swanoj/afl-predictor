"""Lineup derivation and per-team availability adjustment.

We cannot edit the ingest pipeline, but we don't need to: ``PlayerGameLog``
*is* the realized lineup for every historical match — the set of players with a
log for a match is exactly who took the field. ``players_for_match`` exposes
that, and ``availability_adjustment`` converts the value of a (named) lineup
into a *margin points* adjustment relative to the team's recent baseline
lineup.

OPTIMISM CAVEAT (read this before trusting backtest numbers)
-----------------------------------------------------------
For historical matches the only lineup we have is the *realized* one (who
actually played). Using it to "predict" that same match is slightly optimistic
versus a real betting workflow because:

  * Real pre-game team news is the *named 22*, announced ~1-25 hours out, and
    it is occasionally wrong (late outs, vests, positional sub usage).
  * The realized lineup also silently encodes within-game availability (a
    player who was a late out simply has no log), which a forecaster would not
    have known with certainty.

So backtest correlations here are an UPPER BOUND on the signal a production
system extracts. A production system would source the named 22 (e.g. the AFL
site team announcements, Squiggle/“Footy” feeds, or a scraped lineups source)
and pass it as ``expected_lineup`` — the API is identical, only the input
changes. We do NOT claim a free source is wired in; ``expected_lineup`` is the
hook for it.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Match, PlayerGameLog
from src.micro.player_value import (
    DEFAULT_POINTS_PER_VALUE,
    ValueRecord,
    load_player_values,
    replacement_value,
    value_for,
)

# How many of a team's most recent matches define its "baseline lineup value".
DEFAULT_BASELINE_WINDOW = 12


def players_for_match(session: Session, match_id: int, team: str) -> list[str]:
    """Return the realized lineup (player names) for ``team`` in ``match_id``.

    This is the ground-truth "who played", derived from ``PlayerGameLog``.
    """
    return list(
        session.scalars(
            select(PlayerGameLog.player_name).where(
                PlayerGameLog.match_id == match_id,
                PlayerGameLog.team == team,
            )
        ).all()
    )


def lineup_value(
    player_names: list[str],
    team: str,
    values: dict[tuple[str, str], ValueRecord],
    replacement: float,
) -> float:
    """Sum of player values for a named lineup (unknown players -> replacement)."""
    return float(sum(value_for(values, name, team, replacement) for name in player_names))


def _prior_matches(session: Session, team: str, year: int, round_: int) -> list[Match]:
    """Completed matches for ``team`` strictly before (year, round)."""
    matches = session.scalars(
        select(Match)
        .where(
            Match.complete == True,  # noqa: E712
            (Match.home_team == team) | (Match.away_team == team),
        )
        .order_by(Match.year, Match.round, Match.date)
    ).all()
    out = []
    for m in matches:
        if (m.year, m.round) < (year, round_):
            out.append(m)
    return out


def team_baseline_value(
    session: Session,
    team: str,
    year: int,
    round_: int,
    values: dict[tuple[str, str], ValueRecord],
    replacement: float,
    *,
    window: int = DEFAULT_BASELINE_WINDOW,
) -> float | None:
    """The team's baseline lineup value: mean realized lineup value over its
    most recent ``window`` completed matches before (year, round).

    Returns ``None`` if there is no prior history (cold start).
    """
    priors = _prior_matches(session, team, year, round_)
    if not priors:
        return None
    recent = priors[-window:]
    vals: list[float] = []
    for m in recent:
        names = players_for_match(session, m.id, team)
        if names:
            vals.append(lineup_value(names, team, values, replacement))
    if not vals:
        return None
    return sum(vals) / len(vals)


def availability_adjustment(
    session: Session,
    team: str,
    year: int,
    round_: int,
    expected_lineup: list[str] | None = None,
    *,
    values: dict[tuple[str, str], ValueRecord] | None = None,
    points_per_value: float = DEFAULT_POINTS_PER_VALUE,
    window: int = DEFAULT_BASELINE_WINDOW,
) -> float:
    """Points adjustment for ``team`` vs its baseline-strength lineup.

    Positive  -> the (expected) lineup is STRONGER than the team's recent norm
                 (e.g. stars returning) and the team should be favoured more.
    Negative  -> WEAKER than normal (key outs) -> shade the line against them.

    Parameters
    ----------
    expected_lineup:
        The named lineup to evaluate. If ``None`` we fall back to the REALIZED
        lineup from ``PlayerGameLog`` for the (team, year, round) match — this
        is the legitimate-but-optimistic backtest mode (see module docstring).
        In production, pass the named 22 here.
    values:
        Precomputed ``(player, team) -> ValueRecord`` map. Defaults to the
        walk-forward ``PlayerValue`` snapshot for ``year`` (games before
        ``year``), which is leakage-free for predicting that season.
    points_per_value:
        Fitted fantasy-points -> margin-points scalar.

    Returns the margin-points adjustment (0.0 on cold start / no info).
    """
    if values is None:
        values = load_player_values(session, year)
    replacement = replacement_value(values)

    if expected_lineup is None:
        expected_lineup = _realized_lineup(session, team, year, round_)
    if not expected_lineup:
        return 0.0

    current = lineup_value(expected_lineup, team, values, replacement)
    baseline = team_baseline_value(
        session, team, year, round_, values, replacement, window=window
    )
    if baseline is None:
        return 0.0

    return (current - baseline) * points_per_value


def _realized_lineup(
    session: Session, team: str, year: int, round_: int
) -> list[str]:
    """Look up the realized lineup for a (team, year, round) via the Match row."""
    match = session.scalars(
        select(Match).where(
            Match.year == year,
            Match.round == round_,
            (Match.home_team == team) | (Match.away_team == team),
        )
    ).first()
    if match is None:
        return []
    return players_for_match(session, match.id, team)
