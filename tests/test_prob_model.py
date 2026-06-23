"""Tests for the direct win-probability models (logistic + GBM)."""

import numpy as np
import pandas as pd

from src.macro.prob_model import (
    TeamWinProbGBM,
    TeamWinProbLogistic,
    build_win_prob_model,
)
from src.macro.team_model import FEATURE_COLUMNS


def _synthetic_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A separable-ish dataset where home win depends on elo_diff + noise."""
    rng = np.random.default_rng(seed)
    elo_diff = rng.normal(0, 120, n)
    i50_diff = rng.normal(0, 5, n)
    cp_diff = rng.normal(0, 20, n)
    form_diff = rng.normal(0, 0.4, n)
    score_diff = rng.normal(0, 15, n)
    logit = 0.012 * elo_diff + 0.05 * score_diff + 0.3 * form_diff
    p = 1.0 / (1.0 + np.exp(-logit))
    home_win = (rng.random(n) < p).astype(float)
    return pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "i50_diff": i50_diff,
            "cp_diff": cp_diff,
            "home_advantage": 1.0,
            "form_diff": form_diff,
            "score_diff": score_diff,
            "home_win": home_win,
        }
    )


def test_factory_returns_expected_types():
    assert isinstance(build_win_prob_model("logistic"), TeamWinProbLogistic)
    assert isinstance(build_win_prob_model("gbm"), TeamWinProbGBM)


def test_factory_rejects_unknown():
    try:
        build_win_prob_model("nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown kind")


def test_logistic_uses_canonical_features():
    m = TeamWinProbLogistic()
    assert m.feature_columns == FEATURE_COLUMNS


def test_logistic_fit_predict_probabilities_in_unit_interval():
    df = _synthetic_frame()
    m = TeamWinProbLogistic().fit_frame(df, df["home_win"].to_numpy())
    p = m.predict_proba_frame(df)
    assert p.shape == (len(df),)
    assert np.all(p > 0.0) and np.all(p < 1.0)


def test_logistic_learns_signal_better_than_chance():
    df = _synthetic_frame(seed=1)
    train = df.iloc[:400]
    test = df.iloc[400:]
    m = TeamWinProbLogistic().fit_frame(train, train["home_win"].to_numpy())
    p = m.predict_proba_frame(test)
    brier = np.mean((p - test["home_win"].to_numpy()) ** 2)
    # A constant 0.5 predictor scores 0.25; signal should beat that.
    assert brier < 0.24


def test_gbm_fit_predict_probabilities_in_unit_interval():
    df = _synthetic_frame(seed=2)
    m = TeamWinProbGBM().fit_frame(df, df["home_win"].to_numpy())
    p = m.predict_proba_frame(df)
    assert np.all(p > 0.0) and np.all(p < 1.0)


def test_predict_proba_before_fit_raises():
    df = _synthetic_frame()
    m = TeamWinProbLogistic()
    try:
        m.predict_proba_frame(df)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when predicting before fit")
