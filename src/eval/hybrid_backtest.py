"""Hybrid model backtesting."""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Match, MatchEnvironment
from src.eval.backtest import BacktestReport, save_backtest_result
from src.eval.metrics import brier_score, log_loss, mae
from src.ingest.team_features import compute_rolling_stats
from src.macro.elo import EloEngine
from src.macro.environment import from_team_stats
from src.macro.team_model import TeamMarginModel
from src.micro.agent import build_player_baselines, get_team_roster
from src.micro.simulator import HybridSimulator


def walk_forward_hybrid(
    session: Session,
    season: int,
    n_sims: int = 2000,
    seed: int = 42,
) -> BacktestReport:
    """Backtest hybrid Monte Carlo model on a season.

    A fixed ``seed`` makes the Monte Carlo comparison reproducible run-to-run.
    Each match is seeded deterministically off the base seed so games stay
    independent while the whole backtest is repeatable.
    """
    train_seasons = [s for s in range(season - 6, season) if s >= 2018]
    ridge = TeamMarginModel()
    if train_seasons:
        ridge.fit(session, train_seasons)

    build_player_baselines(session, min_games=5, max_year=season - 1)

    elo = EloEngine()
    pre = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year < season)  # noqa: E712
        .order_by(Match.year, Match.round)
    ).all()
    for m in pre:
        if m.home_score is None or m.away_score is None:
            continue
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    test = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year == season)  # noqa: E712
        .order_by(Match.round)
    ).all()

    probs, outcomes, margin_preds, margin_actuals = [], [], [], []

    for idx, m in enumerate(test):
        if m.home_score is None or m.away_score is None:
            continue

        home_roll = compute_rolling_stats(session, m.home_team, m.year, m.round)
        away_roll = compute_rolling_stats(session, m.away_team, m.year, m.round)
        env = from_team_stats(home_roll, away_roll, match_id=m.id)

        home_roster = get_team_roster(session, m.home_team, before_year=m.year, before_round=m.round)
        away_roster = get_team_roster(session, m.away_team, before_year=m.year, before_round=m.round)

        th, ta = ridge.predict_scores(session, m, elo)
        sim = HybridSimulator(env, home_roster, away_roster, th, ta)
        result = sim.run(n_sims=n_sims, seed=seed + idx)

        probs.append(result.home_win_prob)
        outcomes.append(1.0 if m.home_score > m.away_score else 0.0)
        margin_preds.append(result.median_margin)
        margin_actuals.append(m.home_score - m.away_score)

        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    probs_arr = np.array(probs)
    outcomes_arr = np.array(outcomes)

    return BacktestReport(
        model_version="hybrid_v1",
        season=season,
        brier=brier_score(probs_arr, outcomes_arr),
        mae_margin=mae(np.array(margin_preds), np.array(margin_actuals)),
        log_loss_val=log_loss(probs_arr, outcomes_arr),
        n_games=len(probs),
        home_win_accuracy=float(np.mean((probs_arr >= 0.5) == outcomes_arr)),
    )


def run_hybrid_backtest(session: Session, season: int = 2024) -> BacktestReport:
    report = walk_forward_hybrid(session, season)
    save_backtest_result(session, report)
    return report
