"""Tests for conformal prediction intervals."""

import json
from pathlib import Path

import pytest

from src.predict.conformal import clear_conformal_cache, get_conformal_interval


def test_get_conformal_interval_bounds(tmp_path: Path):
    cal_path = tmp_path / "conformal.json"
    cal_path.write_text(
        json.dumps(
            {
                "coverage": 0.9,
                "residual_quantile": 0.4,
                "by_year": {"2024": {"residual_quantile": 0.35, "n": 200}},
            }
        ),
        encoding="utf-8",
    )
    clear_conformal_cache()

    interval = get_conformal_interval(55.0, 2024, conformal_path=cal_path)
    assert interval["coverage"] == 0.9
    assert interval["lower"] == pytest.approx(20.0)
    assert interval["upper"] == pytest.approx(90.0)

    # Falls back to global quantile for unknown year.
    interval2 = get_conformal_interval(0.55, 2018, conformal_path=cal_path)
    assert interval2["lower"] == pytest.approx(15.0)
    assert interval2["upper"] == pytest.approx(95.0)

    clear_conformal_cache()


def test_conformal_clips_at_zero_and_hundred(tmp_path: Path):
    cal_path = tmp_path / "conformal.json"
    cal_path.write_text(
        json.dumps({"coverage": 0.9, "residual_quantile": 0.6}),
        encoding="utf-8",
    )
    clear_conformal_cache()

    low = get_conformal_interval(5.0, 2024, conformal_path=cal_path)
    assert low["lower"] == 0.0

    high = get_conformal_interval(95.0, 2024, conformal_path=cal_path)
    assert high["upper"] == 100.0

    clear_conformal_cache()
