"""Tests for evaluation metrics."""

import numpy as np

from src.eval.metrics import brier_score, log_loss, mae


def test_brier_perfect():
    probs = np.array([1.0, 0.0, 1.0])
    outcomes = np.array([1.0, 0.0, 1.0])
    assert brier_score(probs, outcomes) == 0.0


def test_brier_worst():
    probs = np.array([0.0, 1.0])
    outcomes = np.array([1.0, 0.0])
    assert brier_score(probs, outcomes) == 1.0


def test_mae():
    assert mae(np.array([10, 20]), np.array([12, 18])) == 2.0


def test_log_loss():
    probs = np.array([0.9, 0.1])
    outcomes = np.array([1.0, 0.0])
    assert log_loss(probs, outcomes) < 0.5
