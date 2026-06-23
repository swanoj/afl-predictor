"""Tests for Elo rating system."""

import math

from src.macro.elo import EloEngine, expected_score, margin_multiplier, update_elo


def test_expected_score_equal_teams():
    assert abs(expected_score(1500, 1500) - 0.5) < 0.01


def test_expected_score_favorite():
    p = expected_score(1600, 1400)
    assert p > 0.5


def test_update_elo_home_win():
    r_home, r_away = update_elo(1500, 1500, margin=30)
    assert r_home > 1500
    assert r_away < 1500


def test_update_elo_upset():
    r_home, r_away = update_elo(1600, 1400, margin=-20)
    assert r_home < 1600
    assert r_away > 1400


def test_margin_multiplier():
    m = margin_multiplier(50, 100)
    assert m > 0


def test_elo_engine_predict():
    engine = EloEngine()
    pred = engine.predict("TeamA", "TeamB")
    assert "home_win_prob" in pred
    assert 0 <= pred["home_win_prob"] <= 1


def test_elo_engine_process():
    engine = EloEngine()
    engine.process_match("TeamA", "TeamB", 100, 80)
    assert engine.get("TeamA") > EloEngine().get("TeamA")
