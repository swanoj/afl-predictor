"""Tests for ProbabilityCalibrator (isotonic + Platt/sigmoid)."""

import numpy as np

from src.eval.metrics import expected_calibration_error
from src.macro.calibration_model import ProbabilityCalibrator, calibrate


def _miscalibrated(n: int = 4000, seed: int = 0):
    """Overconfident raw probs: true prob is a shrunk version of the raw one."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.01, 0.99, n)
    # True probability pulled toward 0.5 -> raw probs are overconfident.
    true_p = 0.5 + 0.6 * (raw - 0.5)
    outcomes = (rng.random(n) < true_p).astype(float)
    return raw, outcomes


def test_isotonic_outputs_are_probabilities():
    raw, out = _miscalibrated()
    cal = ProbabilityCalibrator(method="isotonic").fit(raw, out)
    p = cal.transform(raw)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_sigmoid_outputs_are_probabilities():
    raw, out = _miscalibrated(seed=1)
    cal = ProbabilityCalibrator(method="sigmoid").fit(raw, out)
    p = cal.transform(raw)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_isotonic_reduces_ece_on_overconfident_probs():
    raw, out = _miscalibrated(seed=2)
    ece_before = expected_calibration_error(raw, out)
    cal = ProbabilityCalibrator(method="isotonic").fit(raw, out)
    ece_after = expected_calibration_error(cal.transform(raw), out)
    assert ece_after < ece_before


def test_sigmoid_reduces_ece_on_overconfident_probs():
    raw, out = _miscalibrated(seed=3)
    ece_before = expected_calibration_error(raw, out)
    cal = ProbabilityCalibrator(method="sigmoid").fit(raw, out)
    ece_after = expected_calibration_error(cal.transform(raw), out)
    assert ece_after < ece_before


def test_isotonic_is_monotone_nondecreasing():
    raw, out = _miscalibrated(seed=4)
    cal = ProbabilityCalibrator(method="isotonic").fit(raw, out)
    grid = np.linspace(0.02, 0.98, 50)
    mapped = cal.transform(grid)
    assert np.all(np.diff(mapped) >= -1e-9)


def test_transform_before_fit_raises():
    cal = ProbabilityCalibrator(method="isotonic")
    try:
        cal.transform(np.array([0.5]))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when transforming before fit")


def test_unknown_method_raises():
    try:
        ProbabilityCalibrator(method="bogus").fit(np.array([0.5]), np.array([1.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")


def test_calibrate_helper_fits():
    raw, out = _miscalibrated(seed=5)
    cal = calibrate("isotonic", raw, out)
    assert cal.fitted
