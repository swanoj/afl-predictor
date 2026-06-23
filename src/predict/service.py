"""Unified prediction service.

Headline **win probability** is produced by a logistic-regression win-prob model
passed through a sigmoid (Platt) calibrator — the best-calibrated, most
trustworthy model in the Phase B/C backtests (2024 Brier 0.2143, ECE 0.082).
Predicted **margin & scores** come from the Ridge ``TeamMarginModel`` (a useful,
well-behaved point estimate). The ``HybridSimulator`` is invoked solely for
player-level disposal/goal projections.

The Elo engine, fitted Ridge model, fitted logistic model and its calibrator are
all memoised per training-year so repeated calls within a process are fast. The
calibrated-logistic fit reuses the leak-free recipe in
``src.eval.backtest.fit_calibrated_winprob`` (base model trained on all prior
seasons; calibrator fit on a held-out pre-target slice — never the target's
season).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import DEFAULT_SIMS
from src.eval.backtest import fit_calibrated_winprob
from src.ingest.team_features import compute_rolling_stats
from src.macro.calibration_model import ProbabilityCalibrator
from src.macro.elo import EloEngine, build_elo_from_history
from src.macro.environment import EnvironmentState, from_team_stats
from src.macro.prob_model import TeamWinProbLogistic
from src.macro.team_model import (
    DEFAULT_MARGIN_SIGMA,
    TeamMarginModel,
    compute_match_features,
)
from src.micro.agent import PlayerAgent, build_player_baselines, get_team_roster
from src.micro.simulator import HybridSimulator
from src.predict.serialize import (
    WIN_PROB_SOURCE,
    stored_to_item,
    upsert_stored_prediction,
)

# Win-prob model configuration: logistic base + sigmoid (Platt) calibration.
WIN_PROB_KIND = "logistic"
WIN_PROB_CALIB_METHOD = "sigmoid"


class _ModelBundle:
    """Per-training-year set of fitted models used to score matches."""

    def __init__(
        self,
        elo: EloEngine,
        ridge: TeamMarginModel,
        logistic: TeamWinProbLogistic | None,
        calibrator: ProbabilityCalibrator | None,
    ) -> None:
        self.elo = elo
        self.ridge = ridge
        self.logistic = logistic
        self.calibrator = calibrator

    @property
    def win_prob_source(self) -> str:
        if self.logistic is not None and self.calibrator is not None:
            return WIN_PROB_SOURCE
        # Fallback (only when there aren't enough prior seasons to calibrate).
        return "ridge"


# Module-level memo: training-year -> fitted model bundle.
_MODEL_CACHE: dict[int, _ModelBundle] = {}
_BASELINES_BUILT = False


def clear_cache() -> None:
    """Reset the module-level model/baseline memo (useful for tests)."""
    global _BASELINES_BUILT
    _MODEL_CACHE.clear()
    _BASELINES_BUILT = False


def _ensure_baselines(session: Session, *, min_games: int = 5) -> None:
    global _BASELINES_BUILT
    if _BASELINES_BUILT:
        return
    build_player_baselines(session, min_games=min_games)
    # autoflush is off on these sessions; flush so a subsequent roster query
    # sees freshly-added baselines within the same transaction.
    session.flush()
    _BASELINES_BUILT = True


def _get_models(session: Session, year: int) -> _ModelBundle:
    """Return a cached bundle of fitted models for predicting matches in ``year``.

    * Elo — built in-memory (``persist=False``) from full completed history.
    * Ridge — fit on the prior six seasons (>= 2018) for margin & scores.
    * Logistic + sigmoid calibrator — the headline win-prob model, fit with the
      leak-free recipe shared with the backtest harness (calibrator fit on a
      held-out pre-``year`` slice, never on ``year`` itself).
    """
    cached = _MODEL_CACHE.get(year)
    if cached is not None:
        return cached

    elo = build_elo_from_history(session, persist=False)

    ridge = TeamMarginModel()
    ridge.fit(session, list(range(max(2018, year - 6), year)))

    try:
        logistic, calibrator = fit_calibrated_winprob(
            session, year, WIN_PROB_KIND, WIN_PROB_CALIB_METHOD
        )
    except ValueError:
        # Too few prior seasons to calibrate; fall back to Ridge-derived prob.
        logistic, calibrator = None, None

    bundle = _ModelBundle(elo, ridge, logistic, calibrator)
    _MODEL_CACHE[year] = bundle
    return bundle


def _calibrated_win_prob(
    session: Session, match, bundle: _ModelBundle
) -> float | None:
    """Home-win probability from the calibrated logistic model, or ``None`` if
    the model isn't available (insufficient training seasons)."""
    if bundle.logistic is None or bundle.calibrator is None:
        return None
    feats = compute_match_features(session, match, bundle.elo)
    X = pd.DataFrame([feats])
    raw = bundle.logistic.predict_proba_frame(X)
    calibrated = bundle.calibrator.transform(raw)
    return float(calibrated[0])


def _normal_margin_histogram(
    mean: float, sigma: float, n_sims: int, bins: int = 20
) -> list[dict[str, float]]:
    """Analytic margin histogram from a Normal(mean, sigma).

    Counts are derived from the normal CDF over each bin and scaled by
    ``n_sims`` so the shape matches the simulation-based histogram the frontend
    previously consumed, but is now deterministic.
    """
    sigma = sigma if sigma > 1e-6 else DEFAULT_MARGIN_SIGMA
    lo = mean - 4.0 * sigma
    hi = mean + 4.0 * sigma
    edges = np.linspace(lo, hi, bins + 1)
    cdf = norm.cdf(edges, loc=mean, scale=sigma)
    counts = np.diff(cdf) * n_sims
    return [
        {
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "count": int(round(counts[i])),
        }
        for i in range(bins)
    ]


def _player_projections(
    sim_result, home_roster: list[PlayerAgent], away_roster: list[PlayerAgent]
) -> dict[str, dict[str, dict[str, float]]]:
    """Disposal p10/p50/p90 ranges (from the simulator) enriched with per-player
    goal expectations (the goal lambda the simulator drives scoring with)."""
    goal_lambda = {
        a.player_name: float(a.goal_lambda) for a in (*home_roster, *away_roster)
    }

    def _merge(stats: dict[str, dict[str, float]] | None) -> dict[str, dict[str, float]]:
        merged: dict[str, dict[str, float]] = {}
        for name, disp in (stats or {}).items():
            merged[name] = {
                "p10": disp.get("p10", 0.0),
                "p50": disp.get("p50", 0.0),
                "p90": disp.get("p90", 0.0),
                "goal_exp": goal_lambda.get(name, 0.0),
            }
        return merged

    return {
        "home": _merge(sim_result.player_home_disposals),
        "away": _merge(sim_result.player_away_disposals),
    }


def _macro_prediction(
    session: Session, match, bundle: _ModelBundle
) -> dict[str, float]:
    """Compute the macro headline numbers (win prob from calibrated logistic;
    margin & scores from Ridge) for a single match. No player simulation."""
    ridge = bundle.ridge
    margin = ridge.predict_margin(session, match, bundle.elo)
    home_score, away_score = ridge.predict_scores(session, match, bundle.elo)
    sigma = ridge.residual_std if ridge.residual_std > 1e-6 else DEFAULT_MARGIN_SIGMA

    home_win_prob = _calibrated_win_prob(session, match, bundle)
    if home_win_prob is None:
        home_win_prob = ridge.predict_win_prob(margin)

    return {
        "margin": float(margin),
        "home_score": float(home_score),
        "away_score": float(away_score),
        "sigma": float(sigma),
        "home_win_prob": float(home_win_prob),
    }


def predict_match(
    session: Session,
    match,
    *,
    n_sims: int = DEFAULT_SIMS,
    env_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Produce a unified prediction for ``match``.

    Headline win probability comes from the calibrated logistic model; margin
    and scores come from the Ridge model; player projections come from the
    Monte Carlo simulator.

    ``env_overrides`` (pressure_index / territory_tilt / home_momentum /
    away_momentum) only affect the *player projections*: the macro models are
    feature-driven (rolling team form + Elo), not the EnvironmentState, so
    overrides leave the win probability, margin and scores unchanged. This keeps
    the headline prediction coherent with the models that actually drive it.
    """
    _ensure_baselines(session)
    bundle = _get_models(session, match.year)

    home_roll = compute_rolling_stats(session, match.home_team, match.year, match.round)
    away_roll = compute_rolling_stats(session, match.away_team, match.year, match.round)
    env: EnvironmentState = from_team_stats(home_roll, away_roll, match_id=match.id)

    if env_overrides:
        for key in (
            "pressure_index",
            "territory_tilt",
            "home_momentum",
            "away_momentum",
        ):
            val = env_overrides.get(key)
            if val is not None:
                setattr(env, key, val)

    # --- Headline prediction: calibrated logistic (win prob) + Ridge (margin) ---
    macro = _macro_prediction(session, match, bundle)
    margin = macro["margin"]
    home_score = macro["home_score"]
    away_score = macro["away_score"]
    sigma = macro["sigma"]
    home_win_prob = macro["home_win_prob"]
    away_win_prob = 1.0 - home_win_prob

    # --- Player projections: SIMULATOR (win prob / scores ignored) ---
    home_roster = get_team_roster(
        session, match.home_team, before_year=match.year, before_round=match.round
    )
    away_roster = get_team_roster(
        session, match.away_team, before_year=match.year, before_round=match.round
    )
    sim = HybridSimulator(env, home_roster, away_roster, home_score, away_score)
    sim_result = sim.run(n_sims=n_sims)

    p95_margin = float(margin + norm.ppf(0.95) * sigma)

    return {
        "match_id": match.id,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "environment": env.to_dict(),
        # Backward-compatible keys (win probs as 0-100 percentages).
        "home_win_prob": home_win_prob * 100,
        "away_win_prob": away_win_prob * 100,
        "median_home_score": float(home_score),
        "median_away_score": float(away_score),
        "median_margin": float(home_score - away_score),
        "std_margin": float(sigma),
        "p95_margin": p95_margin,
        "margin_histogram": _normal_margin_histogram(margin, sigma, n_sims),
        "player_projections": _player_projections(sim_result, home_roster, away_roster),
        # Provenance so consumers/UX can label sources correctly.
        "win_prob_source": bundle.win_prob_source,
        "margin_source": "ridge",
        "player_projection_source": "monte_carlo",
    }


def _compact_item(session: Session, match, bundle: _ModelBundle) -> dict[str, Any]:
    """Build the compact macro-prediction row for a single ``match``.

    Shared by :func:`predict_round`, :func:`predict_season` and the batch
    builder so the stored rows and the on-the-fly rows are byte-for-byte the same
    shape. No Monte Carlo player simulation is run here.
    """
    macro = _macro_prediction(session, match, bundle)
    home_win_prob = macro["home_win_prob"] * 100
    away_win_prob = 100.0 - home_win_prob
    predicted_winner = (
        match.home_team if home_win_prob >= away_win_prob else match.away_team
    )
    return {
        "match_id": match.id,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "date": match.date.isoformat() if match.date else None,
        "complete": bool(match.complete),
        "home_score": match.home_score,
        "away_score": match.away_score,
        "predicted_winner": predicted_winner,
        "home_win_prob": home_win_prob,
        "away_win_prob": away_win_prob,
        "predicted_margin": float(macro["home_score"] - macro["away_score"]),
        "predicted_home_score": float(macro["home_score"]),
        "predicted_away_score": float(macro["away_score"]),
        "confidence": max(home_win_prob, away_win_prob),
        "win_prob_source": bundle.win_prob_source,
        "margin_source": "ridge",
    }


def predict_round(session: Session, year: int, round_no: int) -> list[dict[str, Any]]:
    """Compact macro predictions for *every* game in ``year`` round ``round_no``.

    Fits Elo/Ridge/logistic ONCE (via the per-year cache) and loops the round's
    matches, skipping the heavy Monte Carlo player simulation for speed. The
    full per-match detail (player projections + histogram) remains available via
    :func:`predict_match`.
    """
    from sqlalchemy import select

    from src.db.models import Match

    bundle = _get_models(session, year)

    matches = session.scalars(
        select(Match)
        .where(Match.year == year, Match.round == round_no)
        .order_by(Match.date, Match.id)
    ).all()

    return [_compact_item(session, match, bundle) for match in matches]


def predict_season(session: Session, year: int) -> list[dict[str, Any]]:
    """Compact macro predictions for *every* match in ``year`` (all rounds).

    Fits Elo/Ridge/logistic ONCE for the season (via the per-year cache) and
    loops every match. Used by the offline batch builder to precompute and store
    predictions; no Monte Carlo player simulation is run.
    """
    from sqlalchemy import select

    from src.db.models import Match

    bundle = _get_models(session, year)

    matches = session.scalars(
        select(Match)
        .where(Match.year == year)
        .order_by(Match.round, Match.date, Match.id)
    ).all()

    return [_compact_item(session, match, bundle) for match in matches]
