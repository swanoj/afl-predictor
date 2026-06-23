"""Tests for the upgraded ratings engine: venue/travel HGA, off/def ratings,
inter-season regression, recency weighting, and backward compatibility."""

from __future__ import annotations

from datetime import date

import pytest

from src.config import ELO_HGA, ELO_START
from src.macro import venues
from src.macro.elo import (
    EloEngine,
    build_elo_from_history,
    compute_home_venue_shares,
)


# --------------------------------------------------------------------------- #
# venues.py
# --------------------------------------------------------------------------- #


def test_haversine_known_distance():
    # Melbourne (MCG) to Perth (Optus) is ~2700 km as the crow flies.
    mel = venues.CITIES["Melbourne"]
    per = venues.CITIES["Perth"]
    d = venues.haversine_km(mel.lat, mel.lon, per.lat, per.lon)
    assert 2600 < d < 2800


def test_haversine_zero_for_same_point():
    assert venues.haversine_km(-37.8, 144.9, -37.8, 144.9) == pytest.approx(0.0, abs=1e-6)


def test_every_db_venue_is_mapped():
    # All 30 venue strings observed in the DB resolve to a city.
    for v in venues.VENUE_CITY:
        assert venues.venue_city(v) is not None


def test_venue_state_resolution():
    assert venues.venue_state("Optus Stadium") == "WA"
    assert venues.venue_state("M.C.G.") == "VIC"
    assert venues.venue_state("Gabba") == "QLD"
    assert venues.venue_state("Totally Unknown Oval") is None


def test_travel_km_home_team_zero_in_own_city():
    # West Coast at Optus (Perth) -> no travel; Geelong (VIC) -> long trip.
    assert venues.travel_km("West Coast", "Optus Stadium") == pytest.approx(0.0, abs=30)
    assert venues.travel_km("Geelong", "Optus Stadium") > 2000


def test_is_interstate():
    assert venues.is_interstate("Geelong", "Optus Stadium") is True
    assert venues.is_interstate("West Coast", "Optus Stadium") is False
    # Both Melbourne clubs at the MCG -> same state.
    assert venues.is_interstate("Carlton", "M.C.G.") is False


def test_travel_unknown_venue_is_zero():
    assert venues.travel_km("Carlton", None) == 0.0
    assert venues.travel_km("Carlton", "Nonexistent Park") == 0.0


def test_effective_hga_rewards_away_travel():
    # Home team in its own city, away team flies interstate -> HGA above base.
    hga = venues.effective_hga("West Coast", "Geelong", "Optus Stadium",
                               base_hga=30.0, travel_coef=0.012)
    assert hga > 30.0


def test_effective_hga_penalises_home_travel():
    # "Home" team forced to play in the opponent's city -> HGA collapses.
    hga = venues.effective_hga("Geelong", "West Coast", "Optus Stadium",
                               base_hga=30.0, travel_coef=0.012)
    assert hga < 30.0


def test_effective_hga_familiarity_scales_base():
    full = venues.effective_hga("Carlton", "Collingwood", "M.C.G.",
                                base_hga=40.0, travel_coef=0.0, home_familiarity=1.0)
    half = venues.effective_hga("Carlton", "Collingwood", "M.C.G.",
                                base_hga=40.0, travel_coef=0.0, home_familiarity=0.5)
    assert full == pytest.approx(40.0)
    assert half == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# Backward compatibility (must not change)
# --------------------------------------------------------------------------- #


def test_default_engine_is_flat():
    e = EloEngine()
    assert e.hga == ELO_HGA
    assert e.hga_mode == "flat"
    # Flat engine ignores venue entirely.
    assert e.hga_for("Geelong", "West Coast", "Optus Stadium") == ELO_HGA
    assert e.hga_for("Geelong", "West Coast", None) == ELO_HGA


def test_predict_returns_legacy_keys():
    e = EloEngine()
    pred = e.predict("A", "B")
    assert set(pred) >= {
        "home_win_prob", "away_win_prob", "home_rating", "away_rating", "elo_diff",
    }
    assert pred["home_win_prob"] + pred["away_win_prob"] == pytest.approx(1.0)
    assert pred["home_rating"] == ELO_START
    assert pred["elo_diff"] == pytest.approx(ELO_HGA)  # equal ratings + flat HGA


def test_process_match_and_get():
    e = EloEngine()
    e.process_match("A", "B", 100, 80)
    assert e.get("A") > ELO_START
    assert e.get("B") < ELO_START
    assert e.get("NeverSeen") == ELO_START


def test_predict_unchanged_by_venue_arg_in_flat_mode():
    e = EloEngine()
    a = e.predict("A", "B")
    b = e.predict("A", "B", venue="Optus Stadium")
    assert a == b


# --------------------------------------------------------------------------- #
# Venue-aware HGA mode
# --------------------------------------------------------------------------- #


def test_venue_mode_changes_hga_with_travel():
    e = EloEngine.upgraded()
    home_hosting = e.hga_for("West Coast", "Geelong", "Optus Stadium")
    home_travelling = e.hga_for("Geelong", "West Coast", "Optus Stadium")
    assert home_hosting > home_travelling


def test_venue_familiarity_from_shares_downweights_neutral():
    e = EloEngine(hga_mode="venue", base_hga=40.0, travel_coef=0.0)
    # Pretend Carlton never plays "home" interstate -> neutral SA venue.
    e.home_venue_shares = {"Carlton": {"M.C.G.": 1.0}}
    in_state = e.hga_for("Carlton", "Adelaide", "M.C.G.")
    neutral = e.hga_for("Carlton", "Adelaide", "Adelaide Oval")
    assert in_state == pytest.approx(40.0)
    assert neutral < in_state  # interstate, unfamiliar -> reduced crowd edge


# --------------------------------------------------------------------------- #
# Offensive / defensive (dual) ratings
# --------------------------------------------------------------------------- #


def test_dual_expected_scores_start_at_league_average():
    e = EloEngine(dual_ratings=True, hga_mode="flat")
    home, away = e.expected_scores("A", "B")
    # No history -> both near the league average, home slightly higher.
    assert home > away
    assert abs(home + away - 2 * 89.0) < 1e-6  # symmetric around the baseline


def test_dual_ratings_track_strong_attack():
    e = EloEngine(dual_ratings=True, hga_mode="flat")
    for _ in range(30):
        e.process_match("Strong", "Weak", 120, 60)
    assert e.get_attack("Strong") > e.get_attack("Weak")
    assert e.get_defense("Strong") > e.get_defense("Weak")
    assert e.expected_margin("Strong", "Weak") > 0


def test_dual_disabled_leaves_ratings_empty():
    e = EloEngine()  # dual off by default
    e.process_match("A", "B", 100, 50)
    assert e.attack == {}
    assert e.defense == {}


# --------------------------------------------------------------------------- #
# Inter-season regression to the mean
# --------------------------------------------------------------------------- #


def test_regress_to_mean_pulls_toward_start():
    e = EloEngine(regress_factor=0.25)
    e.ratings = {"A": 1700.0, "B": 1300.0}
    e.regress_to_mean()
    assert e.ratings["A"] == pytest.approx(ELO_START + 200 * 0.75)
    assert e.ratings["B"] == pytest.approx(ELO_START - 200 * 0.75)


def test_regression_fires_on_season_change():
    e = EloEngine(regress_factor=0.5)
    e.process_match("A", "B", 130, 40, season=2020)
    rating_after_2020 = e.get("A")
    assert rating_after_2020 > ELO_START
    # New season -> rating regressed before the first 2021 match is applied.
    e.process_match("A", "B", 90, 88, season=2021)
    # The 2021 game is near-even so the post-match rating should sit well below
    # the 2020 peak because of the regression step.
    assert e.get("A") < rating_after_2020


def test_no_regression_when_factor_zero():
    e = EloEngine(regress_factor=0.0)
    e.ratings = {"A": 1700.0}
    e.regress_to_mean()
    assert e.ratings["A"] == 1700.0


# --------------------------------------------------------------------------- #
# Recency weighting
# --------------------------------------------------------------------------- #


def test_recency_boosts_k_inside_window():
    e = EloEngine(recency_k_mult=2.0, recency_window_days=365,
                  recency_ref_date=date(2024, 6, 1))
    assert e._k_for(date(2024, 5, 1)) == pytest.approx(e.k * 2.0)  # recent
    assert e._k_for(date(2020, 1, 1)) == pytest.approx(e.k)  # old
    assert e._k_for(None) == pytest.approx(e.k)


def test_recency_disabled_by_default():
    e = EloEngine()
    assert e.recency_k_mult == 1.0
    assert e._k_for(date(2024, 1, 1)) == pytest.approx(e.k)


# --------------------------------------------------------------------------- #
# Home-venue share inference
# --------------------------------------------------------------------------- #


class _FakeMatch:
    def __init__(self, home, venue):
        self.home_team = home
        self.venue = venue


def test_compute_home_venue_shares():
    matches = [
        _FakeMatch("Geelong", "Kardinia Park"),
        _FakeMatch("Geelong", "Kardinia Park"),
        _FakeMatch("Geelong", "M.C.G."),
        _FakeMatch("Geelong", None),  # ignored
    ]
    shares = compute_home_venue_shares(matches)
    assert shares["Geelong"]["Kardinia Park"] == pytest.approx(2 / 3)
    assert shares["Geelong"]["M.C.G."] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# build_elo_from_history (DB-backed; skip when no data)
# --------------------------------------------------------------------------- #


def test_build_elo_from_history_default_and_upgraded():
    from src.db.session import SessionLocal, init_db

    init_db()
    session = SessionLocal()
    try:
        from sqlalchemy import select

        from src.db.models import Match

        has_data = session.scalars(
            select(Match).where(Match.complete == True).limit(1)  # noqa: E712
        ).first()
        if has_data is None:
            pytest.skip("No completed matches in DB")

        flat = build_elo_from_history(session, persist=False)
        assert flat.hga_mode == "flat"
        assert len(flat.ratings) > 0

        upgraded = build_elo_from_history(
            session, persist=False, engine=EloEngine.upgraded()
        )
        assert upgraded.hga_mode == "venue"
        assert len(upgraded.ratings) > 0
        assert upgraded.home_venue_shares  # populated from history
        # Dual ratings populated when enabled.
        assert upgraded.attack
    finally:
        session.close()
