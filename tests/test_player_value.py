"""Tests for the player-value model and availability adjustment.

Uses a hermetic in-memory SQLite DB so the suite does not depend on the
populated production database.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Match, PlayerGameLog, PlayerValue
from src.ingest.lineups import (
    availability_adjustment,
    lineup_value,
    players_for_match,
    team_baseline_value,
)
from src.micro.player_value import (
    FANTASY_WEIGHTS,
    ValueRecord,
    compute_value_table,
    fantasy_points,
    load_player_values,
    replacement_value,
    shrink,
    value_for,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _add_match(s, mid, year, rnd, home, away, hs, aw):
    m = Match(
        id=mid,
        squiggle_id=mid,
        year=year,
        round=rnd,
        home_team=home,
        away_team=away,
        home_score=hs,
        away_score=aw,
        complete=True,
    )
    s.add(m)
    return m


def _add_log(s, mid, name, team, opp, year, rnd, **stats):
    s.add(
        PlayerGameLog(
            match_id=mid,
            player_name=name,
            team=team,
            opponent=opp,
            year=year,
            round=rnd,
            **stats,
        )
    )


def _star_line(**over):
    base = dict(kicks=20, handballs=10, marks=8, tackles=4, hit_outs=0, goals=2, behinds=1)
    base.update(over)
    return base


def _scrub_line(**over):
    base = dict(kicks=4, handballs=3, marks=2, tackles=1, hit_outs=0, goals=0, behinds=0)
    base.update(over)
    return base


def _seed(s, seasons=(2022, 2023), star_team="Geelong", weak_team="North Melbourne"):
    """Two teams playing each other across several rounds, each with a stable
    22 where one team has high-value players and the other low-value."""
    mid = 1
    for year in seasons:
        for rnd in range(1, 11):
            home, away = (star_team, weak_team) if rnd % 2 else (weak_team, star_team)
            _add_match(s, mid, year, rnd, home, away, 100, 80)
            for i in range(22):
                _add_log(s, mid, f"{star_team}_p{i}", star_team, weak_team, year, rnd,
                         **_star_line())
                _add_log(s, mid, f"{weak_team}_p{i}", weak_team, star_team, year, rnd,
                         **_scrub_line())
            mid += 1
    s.commit()
    return mid


# ---------------------------------------------------------------------------
# fantasy_points
# ---------------------------------------------------------------------------
def test_fantasy_points_known_value():
    stats = dict(kicks=10, handballs=5, marks=4, tackles=3, hit_outs=0, goals=2, behinds=1)
    expected = (
        10 * FANTASY_WEIGHTS["kicks"]
        + 5 * FANTASY_WEIGHTS["handballs"]
        + 4 * FANTASY_WEIGHTS["marks"]
        + 3 * FANTASY_WEIGHTS["tackles"]
        + 2 * FANTASY_WEIGHTS["goals"]
        + 1 * FANTASY_WEIGHTS["behinds"]
    )
    assert fantasy_points(stats) == pytest.approx(expected)


def test_fantasy_points_disposal_fallback():
    # No kick/handball split -> fall back to disposals * 2.5.
    only_disp = dict(disposals=20, marks=0, tackles=0, hit_outs=0, goals=0, behinds=0)
    assert fantasy_points(only_disp) == pytest.approx(20 * 2.5)


def test_fantasy_points_accepts_objects():
    class L:
        kicks, handballs, marks, tackles, hit_outs, goals, behinds = 10, 5, 4, 3, 0, 2, 1
        disposals = 15
    assert fantasy_points(L()) == fantasy_points(
        dict(kicks=10, handballs=5, marks=4, tackles=3, hit_outs=0, goals=2, behinds=1)
    )


# ---------------------------------------------------------------------------
# shrink
# ---------------------------------------------------------------------------
def test_shrink_pulls_low_sample_toward_replacement():
    # 1 game, replacement 40 -> heavily shrunk toward 40.
    high_one_game = shrink(120.0, games=1, replacement=40.0, shrinkage=8.0)
    high_many = shrink(120.0, games=50, replacement=40.0, shrinkage=8.0)
    assert high_one_game < high_many
    assert abs(high_one_game - 40.0) < abs(high_many - 40.0)


def test_shrink_zero_games_is_replacement():
    assert shrink(999.0, games=0, replacement=37.0, shrinkage=8.0) == 37.0


# ---------------------------------------------------------------------------
# compute_value_table — walk-forward + persistence
# ---------------------------------------------------------------------------
def test_value_table_ranks_star_above_scrub(session):
    _seed(session)
    values = compute_value_table(session, as_of_season=2024, persist=False)
    star = value_for(values, "Geelong_p0", "Geelong", 0.0)
    scrub = value_for(values, "North Melbourne_p0", "North Melbourne", 0.0)
    assert star > scrub


def test_value_table_is_walk_forward(session):
    # Only 2022-2023 seeded. Asking as_of 2023 must ONLY see 2022 games.
    _seed(session, seasons=(2022, 2023))
    v_2023 = compute_value_table(session, as_of_season=2023, persist=False)
    # 2022 had 10 rounds => each player ~10 games before 2023.
    rec = v_2023[("Geelong_p0", "Geelong")]
    assert rec.games_sample == 10
    # as_of 2022 sees nothing prior -> empty table.
    v_2022 = compute_value_table(session, as_of_season=2022, persist=False)
    assert ("Geelong_p0", "Geelong") not in v_2022


def test_value_table_persists_and_loads(session):
    _seed(session)
    compute_value_table(session, as_of_season=2024, persist=True)
    session.commit()
    rows = session.query(PlayerValue).filter_by(as_of_season=2024).all()
    assert len(rows) > 0
    loaded = load_player_values(session, 2024, compute_if_missing=False)
    assert ("Geelong_p0", "Geelong") in loaded


def test_value_for_unknown_returns_replacement():
    values = {("A", "T"): ValueRecord("A", "T", 80.0, 80.0, 30)}
    assert value_for(values, "Ghost", "T", replacement=12.3) == 12.3


# ---------------------------------------------------------------------------
# lineups
# ---------------------------------------------------------------------------
def test_players_for_match(session):
    _seed(session)
    m = session.query(Match).first()
    names = players_for_match(session, m.id, m.home_team)
    assert len(names) == 22
    assert all(m.home_team in n or n.startswith(m.home_team) for n in names)


def test_lineup_value_sums(session):
    _seed(session)
    values = compute_value_table(session, as_of_season=2024, persist=False)
    rep = replacement_value(values)
    full = [f"Geelong_p{i}" for i in range(22)]
    short = full[:11]
    assert lineup_value(full, "Geelong", values, rep) > lineup_value(
        short, "Geelong", values, rep
    )


def test_availability_adjustment_drop_stars_is_negative(session):
    _seed(session)
    values = compute_value_table(session, as_of_season=2024, persist=False)
    # Evaluate a star team in a hypothetical 2024 round with HALF its stars
    # replaced by replacement-level unknowns -> adjustment must be negative.
    weakened = [f"Geelong_p{i}" for i in range(11)] + [f"unknown_{i}" for i in range(11)]
    adj = availability_adjustment(
        session, "Geelong", year=2024, round_=1,
        expected_lineup=weakened, values=values,
    )
    assert adj < 0.0


def test_availability_adjustment_full_strength_near_zero(session):
    _seed(session)
    values = compute_value_table(session, as_of_season=2024, persist=False)
    full = [f"Geelong_p{i}" for i in range(22)]
    adj = availability_adjustment(
        session, "Geelong", year=2024, round_=1,
        expected_lineup=full, values=values,
    )
    # Same as the baseline lineup it has always fielded -> ~0 adjustment.
    assert abs(adj) < 2.0


def test_availability_adjustment_cold_start_is_zero(session):
    _seed(session)
    values = compute_value_table(session, as_of_season=2024, persist=False)
    # A team with no prior history -> no baseline -> 0.0.
    adj = availability_adjustment(
        session, "Brisbane Lions", year=2024, round_=1,
        expected_lineup=["x", "y"], values=values,
    )
    assert adj == 0.0


def test_team_baseline_none_without_history(session):
    _seed(session)
    values = compute_value_table(session, as_of_season=2024, persist=False)
    rep = replacement_value(values)
    assert team_baseline_value(
        session, "Carlton", 2024, 1, values, rep
    ) is None
