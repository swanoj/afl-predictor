"""Recent-match lineups — actual selected teams, not fantasy-ranked guesses."""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect, or_, select
from sqlalchemy.orm import Session

from src.db.models import Match, PlayerGameLog, ServingRecentLineup


def _lineup_from_match(
    session: Session,
    match: Match,
    team: str,
) -> list[str]:
    """Players who appeared for ``team`` in ``match``, starters first by disposals."""
    logs = session.scalars(
        select(PlayerGameLog)
        .where(PlayerGameLog.match_id == match.id, PlayerGameLog.team == team)
        .order_by(PlayerGameLog.disposals.desc(), PlayerGameLog.player_name)
    ).all()
    return [log.player_name for log in logs]


def _latest_completed_match(
    session: Session,
    team: str,
    year: int,
    before_round: int,
) -> Match | None:
    """Most recent completed fixture for ``team`` in ``year`` before ``before_round``."""
    return session.scalars(
        select(Match)
        .where(
            Match.year == year,
            Match.round < before_round,
            Match.complete.is_(True),
            or_(Match.home_team == team, Match.away_team == team),
        )
        .order_by(Match.round.desc(), Match.id.desc())
        .limit(1)
    ).first()


def _serving_lineups_available(session: Session) -> bool:
    bind = session.get_bind()
    if bind is None:
        return False
    return inspect(bind).has_table(ServingRecentLineup.__tablename__)


def recent_match_lineup(
    session: Session,
    team: str,
    year: int,
    before_round: int,
) -> tuple[list[str], dict[str, Any]] | None:
    """Return the 22 from the team's most recent completed match, with metadata."""
    if _serving_lineups_available(session):
        exported = session.scalars(
            select(ServingRecentLineup.player_name)
            .where(
                ServingRecentLineup.team == team,
                ServingRecentLineup.season == year,
            )
            .order_by(ServingRecentLineup.slot.asc())
        ).all()
        if exported:
            meta_row = session.scalars(
                select(ServingRecentLineup)
                .where(
                    ServingRecentLineup.team == team,
                    ServingRecentLineup.season == year,
                )
                .limit(1)
            ).first()
            meta = {
                "lineup_source": "last_match",
                "source_year": meta_row.source_year if meta_row else year,
                "source_round": meta_row.source_round if meta_row else None,
                "source_opponent": meta_row.source_opponent if meta_row else None,
            }
            return list(exported), meta

    for try_year in (year, year - 1):
        if try_year < 2018:
            continue
        br = before_round if try_year == year else 999
        match = _latest_completed_match(session, team, try_year, br)
        if match is None:
            continue
        lineup = _lineup_from_match(session, match, team)
        if not lineup:
            continue
        opponent = (
            match.away_team if match.home_team == team else match.home_team
        )
        return lineup, {
            "lineup_source": "last_match",
            "source_year": match.year,
            "source_round": match.round,
            "source_opponent": opponent,
        }

    return None


def build_serving_recent_lineups(
    session: Session,
    season: int,
    teams: set[str],
) -> list[dict[str, Any]]:
    """Export last-known 22 per team for the serving DB (engine logs only)."""
    payload: list[dict[str, Any]] = []

    for team in sorted(teams):
        result = recent_match_lineup(session, team, season, before_round=999)
        if result is None:
            continue

        lineup, meta = result
        for slot, player_name in enumerate(lineup[:22], start=1):
            payload.append(
                {
                    "team": team,
                    "season": season,
                    "player_name": player_name,
                    "slot": slot,
                    "source_year": meta.get("source_year"),
                    "source_round": meta.get("source_round"),
                    "source_opponent": meta.get("source_opponent"),
                }
            )

    return payload
