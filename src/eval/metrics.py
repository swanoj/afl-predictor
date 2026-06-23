"""Evaluation metrics for AFL predictions."""

from __future__ import annotations

import numpy as np


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier score for binary outcomes (lower is better)."""
    return float(np.mean((probs - outcomes) ** 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """Binary log loss."""
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - actuals)))


def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def calibration_bins(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Reliability curve bins (equal-width over [0, 1])."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        # Include the right edge in the final bin so p == 1.0 is counted.
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        else:
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        result.append(
            {
                "bin_low": float(bins[i]),
                "bin_high": float(bins[i + 1]),
                "predicted_mean": float(probs[mask].mean()),
                "actual_rate": float(outcomes[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return result


def expected_calibration_error(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE).

    Weighted mean over equal-width probability bins of the absolute gap
    between mean predicted probability and observed outcome rate. 0 means
    perfectly calibrated; higher is worse.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(probs)
    if n == 0:
        return 0.0
    bins = calibration_bins(probs, outcomes, n_bins=n_bins)
    ece = 0.0
    for b in bins:
        weight = b["count"] / n
        ece += weight * abs(b["predicted_mean"] - b["actual_rate"])
    return float(ece)


def maximum_calibration_error(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> float:
    """Maximum Calibration Error (MCE): worst-bin calibration gap."""
    bins = calibration_bins(np.asarray(probs, float), np.asarray(outcomes, float), n_bins)
    if not bins:
        return 0.0
    return float(max(abs(b["predicted_mean"] - b["actual_rate"]) for b in bins))
