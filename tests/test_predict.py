"""Tests for the unified prediction service (Phase C).

Verifies the headline win probability comes from the calibrated logistic model,
margin/scores come from the Ridge model, and player projections come from the
Monte Carlo simulator.
"""

import numpy as np
import pytest
from sqlalchemy import select

from src.db.models import Match
from src.db.session import SessionLocal, init_db
from src.predict import clear_cache, predict_match
from src.predict.service import _normal_margin_histogram, _player_projections


def test_normal_histogram_sums_to_n_sims():
    n_sims = 10000
    hist = _normal_margin_histogram(mean=12.0, sigma=37.0, n_sims=n_sims, bins=20)
    assert len(hist) == 20
    total = sum(b["count"] for b in hist)
    # ~99.99% of the mass falls within +/- 4 sigma, so counts ~ n_sims.
    assert n_sims * 0.99 <= total <= n_sims


def test_normal_histogram_centered_on_mean():
    hist = _normal_margin_histogram(mean=0.0, sigma=30.0, n_sims=10000, bins=20)
    peak = max(hist, key=lambda b: b["count"])
    # The modal bin should straddle the mean.
    assert peak["bin_start"] <= 0.0 <= peak["bin_end"]


def test_normal_histogram_handles_zero_sigma():
    hist = _normal_margin_histogram(mean=5.0, sigma=0.0, n_sims=1000, bins=10)
    assert len(hist) == 10
    assert all(np.isfinite(b["bin_start"]) and np.isfinite(b["bin_end"]) for b in hist)


class _FakeAgent:
    def __init__(self, name, goal_lambda):
        self.player_name = name
        self.goal_lambda = goal_lambda


class _FakeSimResult:
    player_home_disposals = {"Alice": {"p10": 10.0, "p50": 20.0, "p90": 30.0}}
    player_away_disposals = {"Bob": {"p10": 5.0, "p50": 15.0, "p90": 25.0}}


def test_player_projections_merge_goal_exp():
    proj = _player_projections(
        _FakeSimResult(),
        home_roster=[_FakeAgent("Alice", 1.5)],
        away_roster=[_FakeAgent("Bob", 0.3)],
    )
    assert proj["home"]["Alice"]["goal_exp"] == 1.5
    assert proj["home"]["Alice"]["p50"] == 20.0
    assert proj["away"]["Bob"]["goal_exp"] == 0.3


def _first_completed_match(session):
    for year in (2024, 2025):
        m = session.scalars(
            select(Match)
            .where(Match.year == year, Match.complete == True, Match.round >= 5)  # noqa: E712
            .order_by(Match.round)
        ).first()
        if m is not None:
            return m
    return None


def test_predict_match_win_prob_from_calibrated_logistic():
    init_db()
    clear_cache()
    session = SessionLocal()
    try:
        match = _first_completed_match(session)
        if match is None:
            pytest.skip("No completed 2024/2025 matches in DB")

        pred = predict_match(session, match, n_sims=2000)

        # Provenance: win prob = calibrated logistic; margin = Ridge.
        assert pred["win_prob_source"] == "logistic+sigmoid"
        assert pred["margin_source"] == "ridge"
        assert pred["player_projection_source"] == "monte_carlo"

        # Win probs are 0-100 and complementary.
        assert 0.0 <= pred["home_win_prob"] <= 100.0
        assert abs(pred["home_win_prob"] + pred["away_win_prob"] - 100.0) < 1e-6

        # The headline win prob is the calibrated logistic, NOT the normal CDF
        # of the Ridge margin — they should generally differ.
        from src.predict.service import _calibrated_win_prob, _get_models

        bundle = _get_models(session, match.year)
        direct = _calibrated_win_prob(session, match, bundle)
        assert direct is not None
        assert abs(pred["home_win_prob"] - direct * 100) < 1e-6

        # Player projections present and structured.
        assert "home" in pred["player_projections"]
        assert "away" in pred["player_projections"]
        assert len(pred["margin_histogram"]) == 20
    finally:
        session.close()


def test_env_overrides_do_not_change_headline():
    init_db()
    clear_cache()
    session = SessionLocal()
    try:
        match = _first_completed_match(session)
        if match is None:
            pytest.skip("No completed 2024/2025 matches in DB")

        base = predict_match(session, match, n_sims=2000)
        overridden = predict_match(
            session,
            match,
            n_sims=2000,
            env_overrides={
                "pressure_index": 0.95,
                "territory_tilt": 0.9,
                "home_momentum": 1.0,
                "away_momentum": -1.0,
            },
        )
        # The macro headline is feature-driven, independent of env overrides.
        assert base["home_win_prob"] == overridden["home_win_prob"]
        assert base["median_margin"] == overridden["median_margin"]
        # Environment dict reflects the overrides.
        assert overridden["environment"]["pressure_index"] == 0.95
    finally:
        session.close()
