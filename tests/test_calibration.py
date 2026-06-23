"""Tests for calibration math (ECE/MCE/reliability) and Squiggle consensus."""

import numpy as np

from src.eval.calibration import reliability_table
from src.eval.metrics import (
    calibration_bins,
    expected_calibration_error,
    maximum_calibration_error,
)
from src.eval.squiggle_benchmark import build_consensus


def test_ece_perfect_when_probs_equal_outcomes():
    # Predictions exactly equal to outcomes -> perfectly calibrated.
    probs = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    outcomes = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert expected_calibration_error(probs, outcomes) == 0.0


def test_ece_perfect_calibration_grouped():
    # Two groups: p=0.7 with 70% positives, p=0.3 with 30% positives.
    rng = np.random.default_rng(0)
    n = 5000
    probs = np.concatenate([np.full(n, 0.7), np.full(n, 0.3)])
    out_hi = (rng.random(n) < 0.7).astype(float)
    out_lo = (rng.random(n) < 0.3).astype(float)
    outcomes = np.concatenate([out_hi, out_lo])
    ece = expected_calibration_error(probs, outcomes)
    assert ece < 0.02  # close to zero up to sampling noise


def test_ece_worst_case_is_one():
    probs = np.array([0.0, 0.0, 0.0])
    outcomes = np.array([1.0, 1.0, 1.0])
    assert expected_calibration_error(probs, outcomes) == 1.0


def test_ece_known_value():
    # All preds 0.6 in one bin, observed rate 0.5 -> ECE = |0.6-0.5| = 0.1.
    probs = np.full(10, 0.6)
    outcomes = np.array([1.0] * 5 + [0.0] * 5)
    assert abs(expected_calibration_error(probs, outcomes) - 0.1) < 1e-9


def test_ece_empty_is_zero():
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_mce_at_least_ece():
    rng = np.random.default_rng(1)
    probs = rng.random(500)
    outcomes = (rng.random(500) < probs).astype(float)
    ece = expected_calibration_error(probs, outcomes)
    mce = maximum_calibration_error(probs, outcomes)
    assert mce >= ece - 1e-9


def test_calibration_bins_includes_prob_one():
    # p == 1.0 must land in the final bin, not be dropped.
    probs = np.array([1.0, 1.0])
    outcomes = np.array([1.0, 0.0])
    bins = calibration_bins(probs, outcomes, n_bins=10)
    assert sum(b["count"] for b in bins) == 2


def test_reliability_table_gap_sign():
    # Overconfident: predict 0.9 but only 50% win -> negative gap.
    probs = np.full(10, 0.9)
    outcomes = np.array([1.0] * 5 + [0.0] * 5)
    table = reliability_table(probs, outcomes)
    assert len(table) == 1
    assert table[0]["gap"] < 0  # actual_rate - predicted_mean


def test_build_consensus_mean_excludes_meta():
    tips = [
        {"gameid": 1, "source": "ModelA", "home_win_prob": 0.6, "round": 1},
        {"gameid": 1, "source": "ModelB", "home_win_prob": 0.8, "round": 1},
        # Meta-aggregators should be excluded from the consensus mean.
        {"gameid": 1, "source": "Squiggle", "home_win_prob": 0.99, "round": 1},
        {"gameid": 1, "source": "Aggregate", "home_win_prob": 0.01, "round": 1},
    ]
    cons = build_consensus(tips)
    assert set(cons) == {1}
    assert abs(cons[1]["prob"] - 0.7) < 1e-9
    assert cons[1]["n_sources"] == 2
    assert cons[1]["round"] == 1


def test_build_consensus_single_source():
    tips = [
        {"gameid": 1, "source": "ModelA", "home_win_prob": 0.6, "round": 1},
        {"gameid": 1, "source": "Squiggle", "home_win_prob": 0.99, "round": 1},
    ]
    cons = build_consensus(tips, source="Squiggle")
    assert abs(cons[1]["prob"] - 0.99) < 1e-9
    assert cons[1]["n_sources"] == 1


def test_build_consensus_skips_missing_probs():
    tips = [
        {"gameid": 2, "source": "ModelA", "home_win_prob": None, "round": 3},
        {"gameid": 2, "source": "ModelB", "home_win_prob": 0.5, "round": 3},
    ]
    cons = build_consensus(tips)
    assert cons[2]["n_sources"] == 1
    assert abs(cons[2]["prob"] - 0.5) < 1e-9
