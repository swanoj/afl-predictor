"""Post-hoc probability calibration for AFL win-probability models.

A base model emits raw home-win probabilities; a calibrator learns a monotone
map from those raw probabilities to better-calibrated ones, fit on a held-out
slice of the *training* data (never the test season — that would leak).

Two standard methods:

* **Platt / sigmoid** — fit a 1-D logistic regression on the logit of the raw
  probability. Smooth, low-variance, assumes a sigmoidal miscalibration shape.
* **Isotonic** — fit a non-parametric monotone (isotonic) regression. More
  flexible, can correct arbitrary monotone distortion, needs more data.

The :class:`ProbabilityCalibrator` wraps either method behind a common
``fit`` / ``transform`` interface operating purely on (raw_prob, outcome)
arrays, so it is agnostic to which base model produced the probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PROB_EPS = 1e-4


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, PROB_EPS, 1.0 - PROB_EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(np.asarray(p, dtype=float))
    return np.log(p / (1.0 - p))


@dataclass
class ProbabilityCalibrator:
    """Monotone recalibration of raw win probabilities.

    Parameters
    ----------
    method:
        ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.
    """

    method: str = "isotonic"
    fitted: bool = False
    _iso: IsotonicRegression | None = None
    _platt: LogisticRegression | None = None

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> "ProbabilityCalibrator":
        raw = np.asarray(raw_probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        if raw.shape != y.shape:
            raise ValueError("raw_probs and outcomes must have the same shape")
        if len(raw) == 0:
            raise ValueError("Cannot fit calibrator on empty data")

        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(raw, y)
        elif self.method == "sigmoid":
            # Platt scaling: logistic regression on the logit of the raw prob.
            self._platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            self._platt.fit(_logit(raw).reshape(-1, 1), y.astype(int))
        else:
            raise ValueError(f"Unknown calibration method: {self.method!r}")

        self.fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ProbabilityCalibrator is not fitted")
        raw = np.asarray(raw_probs, dtype=float)
        if self.method == "isotonic":
            assert self._iso is not None
            return _clip(self._iso.predict(raw))
        assert self._platt is not None
        proba = self._platt.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        return _clip(proba)

    def fit_transform(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
        return self.fit(raw_probs, outcomes).transform(raw_probs)


def calibrate(method: str, raw_probs: np.ndarray, outcomes: np.ndarray) -> ProbabilityCalibrator:
    """Convenience: build and fit a calibrator in one call."""
    return ProbabilityCalibrator(method=method).fit(raw_probs, outcomes)
