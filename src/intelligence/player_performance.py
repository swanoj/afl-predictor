"""Build player performance tables for the serving DB (export-time only)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Match, PlayerGameLog
from src.intelligence.squads import RECENT_SEASON_WINDOW, active_squad_player_names
from src.micro.agent import infer_role

FORM_WINDOW = 5


def _log_rows(
    session: Session,
    team: str,
    player_name: str,
    *,
    min_year: int,
    max_year: int,
    before_round: int | None = None,
) -> list[PlayerGameLog]:
    conditions = [
        PlayerGameLog.team == team,
        PlayerGameLog.player_name == player_name,
        Match.complete.is_(True),
        Match.year >= min_year,
        Match.year <= max_year,
    ]
    if before_round is not None:
        conditions.append(
            (Match.year < max_year)
            | ((Match.year == max_year) & (Match.round < before_round))
        )
    return list(
        session.scalars(
            select(PlayerGameLog)
            .join(Match, PlayerGameLog.match_id == Match.id)
            .where(*conditions)
            .order_by(Match.year, Match.round)
        ).all()
    )


def _avg(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def _profile_from_logs(logs: list[PlayerGameLog]) -> dict[str, Any]:
    if not logs:
        return {}
    form_logs = logs[-FORM_WINDOW:]
    return {
        "role": infer_role(
            _avg([l.hit_outs for l in logs]),
            _avg([l.goals for l in logs]),
            _avg([l.disposals for l in logs]),
        ),
        "disposals_avg": round(_avg([l.disposals for l in logs]), 1),
        "goals_avg": round(_avg([l.goals for l in logs]), 2),
        "tackles_avg": round(_avg([l.tackles for l in logs]), 1),
        "clearances_avg": round(_avg([l.clearances for l in logs]), 1),
        "hit_outs_avg": round(_avg([l.hit_outs for l in logs]), 1),
        "games": len(logs),
        "form_disposals": round(_avg([l.disposals for l in form_logs]), 1),
        "form_goals": round(_avg([l.goals for l in form_logs]), 2),
        "form_games": len(form_logs),
    }


def build_serving_player_profiles(
    session: Session,
    season: int,
    teams: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (profile_rows, opponent_split_rows) for export."""
    min_year = season - RECENT_SEASON_WINDOW
    profiles: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []

    for team in sorted(teams):
        active = active_squad_player_names(session, team, season)
        for player_name in sorted(active):
            logs = _log_rows(
                session, team, player_name, min_year=min_year, max_year=season
            )
            if not logs:
                continue
            stats = _profile_from_logs(logs)
            profiles.append(
                {
                    "team": team,
                    "season": season,
                    "player_name": player_name,
                    **stats,
                }
            )

            by_opponent: dict[str, list[PlayerGameLog]] = defaultdict(list)
            for row in logs:
                by_opponent[row.opponent].append(row)

            for opponent, opp_logs in by_opponent.items():
                splits.append(
                    {
                        "team": team,
                        "season": season,
                        "player_name": player_name,
                        "opponent": opponent,
                        "disposals_avg": round(
                            _avg([l.disposals for l in opp_logs]), 1
                        ),
                        "goals_avg": round(_avg([l.goals for l in opp_logs]), 2),
                        "games": len(opp_logs),
                    }
                )

    return profiles, splits
