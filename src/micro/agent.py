"""Player agent with Beta/Poisson distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import PlayerBaseline, PlayerGameLog
from src.macro.environment import EnvironmentState


ROLE_PRESSURE_SENS = {
    "Ruck": 0.15,
    "Inside Mid": 0.35,
    "Outside Mid": 0.1,
    "Key Forward": -0.25,
    "Forward": -0.15,
    "Key Defender": 0.2,
    "Defender": 0.1,
    "Wing": 0.05,
    "Generic": 0.0,
}

ROLE_TERRITORY_SENS = {
    "Ruck": 0.1,
    "Inside Mid": 0.15,
    "Outside Mid": 0.2,
    "Key Forward": 0.45,
    "Forward": 0.35,
    "Key Defender": 0.05,
    "Defender": 0.05,
    "Wing": 0.15,
    "Generic": 0.1,
}


def infer_role(avg_hit_outs: float, avg_goals: float, avg_disposals: float) -> str:
    if avg_hit_outs >= 12:
        return "Ruck"
    if avg_goals >= 1.8:
        return "Key Forward"
    if avg_goals >= 0.8:
        return "Forward"
    if avg_disposals >= 22 and avg_goals < 0.5:
        return "Inside Mid"
    if avg_disposals >= 18:
        return "Outside Mid"
    if avg_disposals >= 14 and avg_goals < 0.3:
        return "Defender"
    return "Generic"


@dataclass
class PlayerAgent:
    player_name: str
    team: str
    role: str
    disp_floor: int
    disp_p50: int
    disp_ceil: int
    goal_lambda: float
    behind_lambda: float
    alpha: float
    beta: float
    pressure_sensitivity: float
    territory_sensitivity: float

    @classmethod
    def from_baseline(cls, b: PlayerBaseline) -> PlayerAgent:
        range_size = max(1, b.disp_ceil - b.disp_floor)
        norm_avg = (b.disp_p50 - b.disp_floor) / range_size
        kappa = 10.0
        alpha = max(0.1, norm_avg * kappa)
        beta = max(0.1, (1.0 - norm_avg) * kappa)
        return cls(
            player_name=b.player_name,
            team=b.team,
            role=b.role,
            disp_floor=b.disp_floor,
            disp_p50=b.disp_p50,
            disp_ceil=b.disp_ceil,
            goal_lambda=b.goal_lambda,
            behind_lambda=b.behind_lambda,
            alpha=alpha,
            beta=beta,
            pressure_sensitivity=b.pressure_sensitivity,
            territory_sensitivity=b.territory_sensitivity,
        )


def build_player_baselines(
    session: Session, min_games: int = 15, max_year: int | None = None
) -> int:
    """Compute player baselines from historical game logs.

    When ``max_year`` is set, only logs from seasons at or before ``max_year``
    are used, preventing look-ahead leakage during walk-forward backtesting.
    """
    query = select(PlayerGameLog)
    if max_year is not None:
        query = query.where(PlayerGameLog.year <= max_year)
    logs = session.scalars(query).all()
    if not logs:
        return 0

    df = pd.DataFrame(
        [
            {
                "player_name": l.player_name,
                "team": l.team,
                "disposals": l.disposals,
                "goals": l.goals,
                "behinds": l.behinds,
                "hit_outs": l.hit_outs,
            }
            for l in logs
        ]
    )

    count = 0
    for (player, team), group in df.groupby(["player_name", "team"]):
        if len(group) < min_games:
            continue

        disp_floor = int(group["disposals"].quantile(0.1))
        disp_p50 = int(group["disposals"].quantile(0.5))
        disp_ceil = int(group["disposals"].quantile(0.9))
        goal_lambda = float(group["goals"].mean())
        behind_lambda = float(group["behinds"].mean())
        role = infer_role(
            group["hit_outs"].mean(), group["goals"].mean(), group["disposals"].mean()
        )

        existing = session.scalars(
            select(PlayerBaseline).where(
                PlayerBaseline.player_name == player,
                PlayerBaseline.team == team,
            )
        ).first()

        ps = ROLE_PRESSURE_SENS.get(role, 0.0)
        ts = ROLE_TERRITORY_SENS.get(role, 0.1)

        if existing:
            existing.role = role
            existing.disp_floor = disp_floor
            existing.disp_p50 = disp_p50
            existing.disp_ceil = disp_ceil
            existing.goal_lambda = goal_lambda
            existing.behind_lambda = behind_lambda
            existing.games_sample = len(group)
            existing.pressure_sensitivity = ps
            existing.territory_sensitivity = ts
        else:
            session.add(
                PlayerBaseline(
                    player_name=player,
                    team=team,
                    role=role,
                    disp_floor=disp_floor,
                    disp_p50=disp_p50,
                    disp_ceil=disp_ceil,
                    goal_lambda=goal_lambda,
                    behind_lambda=behind_lambda,
                    games_sample=len(group),
                    pressure_sensitivity=ps,
                    territory_sensitivity=ts,
                )
            )
        count += 1

    return count


def get_team_roster(
    session: Session,
    team: str,
    max_players: int = 22,
    before_year: int | None = None,
    before_round: int | None = None,
) -> list[PlayerAgent]:
    """Get roster from baselines, optionally filtered to players active before a round."""
    from src.db.models import Match

    if before_year is not None and before_round is not None:
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
        if active_names:
            baselines = session.scalars(
                select(PlayerBaseline)
                .where(
                    PlayerBaseline.team == team,
                    PlayerBaseline.player_name.in_(active_names),
                )
                .order_by(
                    PlayerBaseline.games_sample.desc(),
                    PlayerBaseline.player_name,
                )
                .limit(max_players)
            ).all()
            if baselines:
                return [PlayerAgent.from_baseline(b) for b in baselines]

    baselines = session.scalars(
        select(PlayerBaseline)
        .where(PlayerBaseline.team == team)
        .order_by(
            PlayerBaseline.games_sample.desc(),
            PlayerBaseline.player_name,
        )
        .limit(max_players)
    ).all()
    return [PlayerAgent.from_baseline(b) for b in baselines]
