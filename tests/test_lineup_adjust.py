"""Tests for numpy-free lineup adjustment helpers."""

from __future__ import annotations

import pytest

from src.intelligence.lineups import (
    expected_lineup,
    players_out_from_injuries,
    roster_candidates_from_values,
)
from src.intelligence.news import InjuryUpdate
from src.predict.lineup_adjust import (
    apply_lineup_adjustment,
    home_away_margin_adjustment,
    lineup_value,
    shift_win_prob,
    team_availability_margin,
)


def _values() -> dict[tuple[str, str], float]:
    team = "Geelong"
    return {(f"p{i}", team): 100.0 - i for i in range(30)}


def test_lineup_value_sums():
    values = _values()
    full = [f"p{i}" for i in range(22)]
    short = full[:11]
    rep = 10.0
    assert lineup_value(full, "Geelong", values, rep) > lineup_value(
        short, "Geelong", values, rep
    )


def test_team_availability_margin_negative_when_weakened():
    values = _values()
    full = [f"p{i}" for i in range(22)]
    weakened = full[:11] + ["unknown"] * 11
    baseline = lineup_value(full, "Geelong", values, 10.0)
    adj = team_availability_margin(weakened, "Geelong", baseline, values)
    assert adj < 0.0


def test_home_away_margin_adjustment_sign():
    values = _values()
    geelong_full = [f"p{i}" for i in range(22)]
    geelong_weak = geelong_full[:11] + ["x"] * 11
    baseline = lineup_value(geelong_full, "Geelong", values, 10.0)

    # Home weakened, away at baseline -> negative home-minus-away adj.
    adj = home_away_margin_adjustment(
        geelong_weak,
        geelong_full,
        "Geelong",
        "Geelong",
        baseline,
        baseline,
        values,
    )
    assert adj < 0.0


def test_shift_win_prob_clamps():
    home, away = shift_win_prob(0.55, margin_adj=-50.0, sensitivity=0.02)
    assert home == pytest.approx(0.05)
    assert away == pytest.approx(0.95)


def test_shift_win_prob_respects_percentage_scale():
    home, away = shift_win_prob(55.0, margin_adj=10.0, sensitivity=0.02)
    assert home == pytest.approx(55.2)
    assert away == pytest.approx(44.8)


def test_shift_win_prob_clamps_percentage_scale():
    home, away = shift_win_prob(55.0, margin_adj=-3000.0, sensitivity=0.02)
    assert home == pytest.approx(5.0)
    assert away == pytest.approx(95.0)


def test_apply_lineup_adjustment_provenance():
    values = _values()
    for i in range(22):
        values[(f"a{i}", "Carlton")] = 100.0 - i
    full_home = [f"p{i}" for i in range(22)]
    full_away = [f"a{i}" for i in range(22)]
    home_baseline = lineup_value(full_home, "Geelong", values, 10.0)
    away_baseline = lineup_value(full_away, "Carlton", values, 10.0)
    result = apply_lineup_adjustment(
        55.0,
        full_home,
        full_away,
        "Geelong",
        "Carlton",
        home_baseline,
        away_baseline,
        values,
    )
    assert result["win_prob_source"] == "logistic+sigmoid+lineup"
    assert result["lineup_margin_adj"] == pytest.approx(0.0, abs=0.01)


def test_expected_lineup_skips_outs():
    roster = [f"p{i}" for i in range(25)]
    out = {"p0", "p1", "p2"}
    lineup = expected_lineup(roster, out, size=22)
    assert len(lineup) == 22
    assert not out.intersection(lineup)


def test_players_out_from_injuries():
    injuries = [
        InjuryUpdate(
            player="Marcus Bontempelli",
            team="Western Bulldogs",
            status="out",
            headline="ruled out",
            url="",
            published=None,
        ),
        InjuryUpdate(
            player="Pat Cripps",
            team="Carlton",
            status="doubtful",
            headline="test",
            url="",
            published=None,
        ),
    ]
    out = players_out_from_injuries(injuries, "Western Bulldogs")
    assert out == {"Marcus Bontempelli"}


def test_roster_candidates_sorted_by_value():
    values = _values()
    roster = roster_candidates_from_values(values, "Geelong")
    assert roster[0] == "p0"
    assert len(roster) == 30
