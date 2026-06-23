"""Conformal prediction intervals for home win probability.

Calibration quantiles are computed offline by ``scripts/build_conformal.py``
from walk-forward backtest residuals and shipped as ``deploy/conformal.json``.
This module is numpy-free so it can run in the lean serving image.

Interval construction (symmetric split conformal on absolute residuals):

    lower = max(0, p - q)
    upper = min(1, p + q)

where ``q`` is the calibrated quantile of ``|outcome - predicted_prob|`` and
``p`` is the model's home win probability in [0, 1].
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import ROOT_DIR

DEFAULT_CONFORMAL_PATH = ROOT_DIR / "deploy" / "conformal.json"
DEFAULT_COVERAGE = 0.9


@lru_cache(maxsize=1)
def _load_calibration(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {
            "coverage": DEFAULT_COVERAGE,
            "residual_quantile": 0.45,
            "by_year": {},
        }
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def clear_conformal_cache() -> None:
    """Reset cached calibration (for tests)."""
    _load_calibration.cache_clear()


def _prob_fraction(home_win_prob: float) -> float:
    """Accept 0-1 or 0-100 scale."""
    if home_win_prob > 1.0:
        return home_win_prob / 100.0
    return home_win_prob


def get_conformal_interval(
    home_win_prob: float,
    year: int,
    *,
    conformal_path: Path | None = None,
) -> dict[str, float]:
    """Return a symmetric conformal interval for home win probability.

    Parameters
    ----------
    home_win_prob:
        Model probability on 0-100 (API) or 0-1 scale.
    year:
        Season used to pick a year-specific quantile when available.

    Returns
    -------
    dict with ``lower``, ``upper`` (0-100 scale) and ``coverage`` (0.9).
    """
    cal = _load_calibration(str(conformal_path or DEFAULT_CONFORMAL_PATH))
    coverage = float(cal.get("coverage", DEFAULT_COVERAGE))

    by_year = cal.get("by_year") or {}
    year_entry = by_year.get(str(year)) or by_year.get(year)
    if isinstance(year_entry, dict) and "residual_quantile" in year_entry:
        q = float(year_entry["residual_quantile"])
    else:
        q = float(cal.get("residual_quantile", 0.45))

    p = _prob_fraction(home_win_prob)
    lower = max(0.0, p - q)
    upper = min(1.0, p + q)

    return {
        "lower": round(lower * 100.0, 1),
        "upper": round(upper * 100.0, 1),
        "coverage": coverage,
    }
