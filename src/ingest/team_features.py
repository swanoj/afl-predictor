"""Team-level feature aggregation from player logs and match data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Match, PlayerGameLog, TeamMatchStats

logger = logging.getLogger(__name__)


@dataclass
class RollingTeamStats:
    team: str
    games: int
    avg_score: float
    avg_conceded: float
    i50_diff: float
    cp_rate: float
    tackle_rate: float
    momentum: float
    win_streak: int


def build_team_stats_from_players(session: Session, match_id: int, team: str, is_home: bool) -> TeamMatchStats:
    """Aggregate player logs into team match stats for a single match."""
    logs = session.scalars(
        select(PlayerGameLog).where(
            PlayerGameLog.match_id == match_id,
            PlayerGameLog.team == team,
        )
    ).all()

    tackles = sum(l.tackles for l in logs)
    clearances = sum(getattr(l, "clearances", 0) for l in logs)
    score = sum(l.goals * 6 + l.behinds for l in logs)

    # REAL summed team stats from AFL Tables player columns.
    # CP summed across players IS the team contested-possession count;
    # summing player inside-50 deliveries is the standard team I50 proxy.
    real_cp = sum(getattr(l, "contested_poss", 0) or 0 for l in logs)
    real_i50 = sum(getattr(l, "inside50", 0) or 0 for l in logs)

    if real_cp > 0:
        contested = float(real_cp)
    else:
        # Genuinely unavailable for this match/season; fall back and log it.
        contested = sum((l.disposals * 0.35 + l.tackles) for l in logs)
        logger.warning(
            "No real contested-possession data for match %s team %s; using proxy",
            match_id,
            team,
        )

    if real_i50 > 0:
        i50_for = float(real_i50)
    else:
        i50_for = max(score / 4.5, len(logs) * 2.5)
        logger.warning(
            "No real inside-50 data for match %s team %s; using proxy",
            match_id,
            team,
        )

    existing = session.scalars(
        select(TeamMatchStats).where(
            TeamMatchStats.match_id == match_id,
            TeamMatchStats.team == team,
        )
    ).first()

    if existing:
        existing.contested_poss = contested
        existing.tackles = tackles
        existing.clearances = clearances
        existing.score = score
        existing.i50_for = i50_for
        existing.is_home = is_home
        return existing

    stat = TeamMatchStats(
        match_id=match_id,
        team=team,
        is_home=is_home,
        score=score,
        contested_poss=contested,
        tackles=tackles,
        clearances=clearances,
        i50_for=i50_for,
    )
    session.add(stat)
    return stat


def compute_rolling_stats(
    session: Session,
    team: str,
    before_year: int,
    before_round: int,
    window: int = 5,
) -> RollingTeamStats:
    """Compute rolling team stats from completed matches before a given round."""
    matches = session.scalars(
        select(Match)
        .where(Match.complete == True)  # noqa: E712
        .where(
            (Match.year < before_year)
            | ((Match.year == before_year) & (Match.round < before_round))
        )
        .order_by(Match.year, Match.round)
    ).all()

    team_games: list[dict] = []
    for m in matches:
        if m.home_team == team:
            margin = (m.home_score or 0) - (m.away_score or 0)
            team_games.append(
                {
                    "score": m.home_score or 0,
                    "conceded": m.away_score or 0,
                    "won": margin > 0,
                    "margin": margin,
                }
            )
        elif m.away_team == team:
            margin = (m.away_score or 0) - (m.home_score or 0)
            team_games.append(
                {
                    "score": m.away_score or 0,
                    "conceded": m.home_score or 0,
                    "won": margin > 0,
                    "margin": margin,
                }
            )

    if not team_games:
        return RollingTeamStats(
            team=team,
            games=0,
            avg_score=80.0,
            avg_conceded=80.0,
            i50_diff=0.0,
            cp_rate=300.0,
            tackle_rate=60.0,
            momentum=0.0,
            win_streak=0,
        )

    recent = team_games[-window:]
    df = pd.DataFrame(recent)

    win_streak = 0
    for g in reversed(team_games):
        if g["won"]:
            win_streak += 1
        else:
            break

    margins = df["margin"].values
    momentum = float(np.clip(np.mean(margins) / 40.0 + win_streak * 0.1, -1, 1))

    # Pull team stats if available (strictly before the prediction cutoff)
    stat_rows = session.scalars(
        select(TeamMatchStats)
        .join(Match)
        .where(
            TeamMatchStats.team == team,
            Match.complete == True,  # noqa: E712
            (Match.year < before_year)
            | ((Match.year == before_year) & (Match.round < before_round)),
        )
        .order_by(Match.year.desc(), Match.round.desc())
        .limit(window)
    ).all()

    if stat_rows:
        i50_diff = float(np.mean([(s.i50_for or 0) - (s.i50_against or 0) for s in stat_rows]))
        cp_rate = float(np.mean([s.contested_poss or 300 for s in stat_rows]))
        tackle_rate = float(np.mean([s.tackles or 60 for s in stat_rows]))
    else:
        i50_diff = float(np.mean(df["score"] - df["conceded"]) / 5)
        cp_rate = 300.0
        tackle_rate = 60.0

    return RollingTeamStats(
        team=team,
        games=len(recent),
        avg_score=float(df["score"].mean()),
        avg_conceded=float(df["conceded"].mean()),
        i50_diff=i50_diff,
        cp_rate=cp_rate,
        tackle_rate=tackle_rate,
        momentum=momentum,
        win_streak=win_streak,
    )
