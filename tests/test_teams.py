"""Tests for team ladder, form, and squad helpers."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Match, PlayerValue
from src.intelligence.teams import (
    TEAM_META,
    compute_ladder,
    team_form,
    team_season_summary,
    team_squad,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _add_match(s, mid, year, rnd, home, away, hs, aw, *, complete=True):
    s.add(
        Match(
            id=mid,
            squiggle_id=mid,
            year=year,
            round=rnd,
            home_team=home,
            away_team=away,
            home_score=hs,
            away_score=aw,
            complete=complete,
        )
    )


def test_team_meta_has_eighteen_teams():
    assert len(TEAM_META) == 18
    assert "Carlton" in TEAM_META
    assert TEAM_META["Carlton"]["abbreviation"] == "CAR"


def test_compute_ladder_orders_by_points(session):
    _add_match(session, 1, 2026, 1, "Carlton", "Collingwood", 90, 70)
    _add_match(session, 2, 2026, 1, "Richmond", "Essendon", 60, 80)
    session.commit()

    ladder = compute_ladder(session, 2026)
    played = [row for row in ladder if row["played"] > 0]
    assert played[0]["team"] == "Essendon"
    assert played[0]["wins"] == 1
    assert played[0]["ladder_points"] == 4
    assert played[1]["team"] == "Carlton"
    assert any(row["team"] == "Collingwood" and row["losses"] == 1 for row in played)


def test_team_form_last_five(session):
    for rnd in range(1, 8):
        _add_match(session, rnd, 2026, rnd, "Carlton", "Richmond", 80 + rnd, 70)
    session.commit()

    form = team_form(session, "Carlton", 2026, before_round=8)
    assert len(form) == 5
    assert all(item["result"] == "W" for item in form)
    assert form[-1]["round"] == 7


def test_team_squad_sorted_by_value(session):
    session.add_all(
        [
            PlayerValue(
                player_name="Star",
                team="Carlton",
                as_of_season=2026,
                value=95.0,
                raw_value=95.0,
                games_sample=20,
            ),
            PlayerValue(
                player_name="Bench",
                team="Carlton",
                as_of_season=2026,
                value=40.0,
                raw_value=40.0,
                games_sample=10,
            ),
        ]
    )
    session.commit()

    squad = team_squad(session, "Carlton", 2026, limit=5)
    assert squad[0]["player_name"] == "Star"
    assert squad[0]["value"] == 95.0


def test_team_season_summary_before_round(session):
    _add_match(session, 1, 2026, 1, "Carlton", "Collingwood", 90, 70)
    _add_match(session, 2, 2026, 2, "Carlton", "Richmond", 50, 80)
    session.commit()

    summary = team_season_summary(session, "Carlton", 2026, before_round=2)
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["played"] == 1
    assert summary["streak"] == "W"
