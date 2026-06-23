"""Expected lineups for upcoming matches (numpy-free serving path).

Roster candidates come from the compact ``player_values`` rows exported into
``deploy/serving.db``. Players flagged ``out`` (or ``omitted`` / ``sidelined``)
in injury headlines are removed; the highest-value remaining candidates fill
the expected 22.

When the full engine DB is available (offline / dev), roster candidates can
also be inferred from recent ``PlayerGameLog`` activity via
``roster_from_game_logs``.
"""

from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import Match, PlayerGameLog, PlayerValue
from src.intelligence.news import InjuryUpdate
from src.predict.lineup_adjust import PlayerValueMap, lineup_value, replacement_level

LINEUP_SIZE = 22

# Injury statuses treated as definite absences for lineup building.
OUT_STATUSES = frozenset({"out", "omitted", "sidelined"})


def load_player_values_map(
    session: Session,
    as_of_season: int,
) -> PlayerValueMap:
    """Load ``(player_name, team) -> value`` from persisted ``PlayerValue`` rows."""
    rows = session.scalars(
        select(PlayerValue).where(PlayerValue.as_of_season == as_of_season)
    ).all()
    if not rows:
        latest = session.scalar(select(func.max(PlayerValue.as_of_season)))
        if latest and latest != as_of_season:
            rows = session.scalars(
                select(PlayerValue).where(PlayerValue.as_of_season == latest)
            ).all()
    return {(r.player_name, r.team): float(r.value) for r in rows}


def teams_in_season(session: Session, year: int) -> set[str]:
    """All club names appearing in ``year`` fixtures."""
    matches = session.scalars(select(Match).where(Match.year == year)).all()
    teams: set[str] = set()
    for m in matches:
        teams.add(m.home_team)
        teams.add(m.away_team)
    return teams


def roster_candidates_from_values(
    values: PlayerValueMap,
    team: str,
) -> list[str]:
    """All known players for ``team``, highest value first."""
    team_entries = [
        (name, val)
        for (name, t), val in values.items()
        if t == team
    ]
    team_entries.sort(key=lambda x: (-x[1], x[0]))
    return [name for name, _ in team_entries]


def roster_from_game_logs(
    session: Session,
    team: str,
    *,
    before_year: int,
    before_round: int,
    pool_size: int = 30,
) -> list[str]:
    """Recent active players from ``PlayerGameLog`` (full engine DB only)."""
    active_names = session.scalars(
        select(PlayerGameLog.player_name)
        .join(Match, PlayerGameLog.match_id == Match.id)
        .where(
            PlayerGameLog.team == team,
            (Match.year < before_year)
            | ((Match.year == before_year) & (Match.round < before_round)),
            Match.year >= before_year - 1,
        )
        .distinct()
    ).all()
    if not active_names:
        return []
    return sorted(active_names)[:pool_size]


def players_out_from_injuries(
    injuries: Sequence[InjuryUpdate],
    team: str,
) -> set[str]:
    """Player names ruled out for ``team`` according to parsed headlines."""
    out: set[str] = set()
    for item in injuries:
        if item.team != team:
            continue
        if item.status.lower() not in OUT_STATUSES:
            continue
        if item.player and item.player != "Squad":
            out.add(item.player)
    return out


def expected_lineup(
    roster_candidates: Sequence[str],
    out_players: set[str],
    *,
    size: int = LINEUP_SIZE,
) -> list[str]:
    """Build the expected ``size`` from candidates minus definite outs."""
    selected: list[str] = []
    out_lower = {p.lower() for p in out_players}
    for name in roster_candidates:
        if name.lower() in out_lower:
            continue
        selected.append(name)
        if len(selected) >= size:
            break
    return selected


def baseline_lineup_value_proxy(
    roster_candidates: Sequence[str],
    team: str,
    values: PlayerValueMap,
    *,
    size: int = LINEUP_SIZE,
    replacement: float | None = None,
) -> float | None:
    """Proxy baseline: value of the team's top ``size`` roster candidates.

    Used in serving when historical realized lineups (``PlayerGameLog``) are
    unavailable. With a full engine DB, prefer ``team_baseline_value`` from
    ``src.ingest.lineups`` instead.
    """
    if not roster_candidates:
        return None
    rep = replacement if replacement is not None else replacement_level(values)
    full_strength = list(roster_candidates[:size])
    if not full_strength:
        return None
    return lineup_value(full_strength, team, values, rep)


def derive_team_lineup(
    session: Session,
    team: str,
    year: int,
    round_: int,
    injuries: Sequence[InjuryUpdate],
    values: PlayerValueMap,
    *,
    replacement: float | None = None,
) -> dict[str, Any]:
    """Expected 22 + metadata for one team in an upcoming match."""
    rep = replacement if replacement is not None else replacement_level(values)

    candidates = roster_candidates_from_values(values, team)
    if not candidates:
        candidates = roster_from_game_logs(
            session, team, before_year=year, before_round=round_
        )

    out_players = players_out_from_injuries(injuries, team)
    lineup = expected_lineup(candidates, out_players)
    baseline = baseline_lineup_value_proxy(candidates, team, values, replacement=rep)

    return {
        "team": team,
        "expected_lineup": lineup,
        "out_players": sorted(out_players),
        "roster_pool_size": len(candidates),
        "lineup_value": lineup_value(lineup, team, values, rep) if lineup else 0.0,
        "baseline_lineup_value": baseline,
    }


def derive_match_lineups(
    session: Session,
    match: Match,
    injuries: Sequence[InjuryUpdate] | None = None,
    *,
    as_of_season: int | None = None,
) -> dict[str, Any]:
    """Expected home/away lineups and value summaries for ``match``."""
    season = as_of_season if as_of_season is not None else match.year
    values = load_player_values_map(session, season)
    injuries = injuries or []

    home = derive_team_lineup(
        session,
        match.home_team,
        match.year,
        match.round,
        injuries,
        values,
    )
    away = derive_team_lineup(
        session,
        match.away_team,
        match.year,
        match.round,
        injuries,
        values,
    )

    return {
        "match_id": match.id,
        "year": match.year,
        "round": match.round,
        "home": home,
        "away": away,
        "values_loaded": len(values),
    }
