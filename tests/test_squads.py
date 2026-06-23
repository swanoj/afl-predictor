"""Tests for current-squad filtering (retired players excluded)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Match, PlayerGameLog, PlayerValue, ServingRoster
from src.intelligence.squads import (
    active_squad_player_names,
    roster_candidates,
    team_squad_rows,
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


def test_active_squad_excludes_retired_players(session):
    _add_match(session, 1, 2025, 1, "Carlton", "Richmond", 90, 70)
    session.add(
        PlayerGameLog(
            match_id=1,
            player_name="Walsh, Sam",
            team="Carlton",
            opponent="Richmond",
            year=2025,
            round=1,
            goals=2,
            disposals=30,
        )
    )
    session.add_all(
        [
            PlayerValue(
                player_name="Walsh, Sam",
                team="Carlton",
                as_of_season=2026,
                value=90.0,
                raw_value=90.0,
                games_sample=50,
            ),
            PlayerValue(
                player_name="Murphy, Marc",
                team="Carlton",
                as_of_season=2026,
                value=85.0,
                raw_value=85.0,
                games_sample=200,
            ),
        ]
    )
    session.commit()

    active = active_squad_player_names(session, "Carlton", 2026, before_round=1)
    assert "Walsh, Sam" in active
    assert "Murphy, Marc" not in active

    values = {("Walsh, Sam", "Carlton"): 90.0, ("Murphy, Marc", "Carlton"): 85.0}
    roster = roster_candidates(session, "Carlton", 2026, 1, values)
    assert roster == ["Walsh, Sam"]
    assert "Murphy, Marc" not in roster


def test_serving_roster_used_on_lean_db(session):
    session.add(
        ServingRoster(
            team="Carlton",
            season=2026,
            player_name="Walsh, Sam",
            value=90.0,
            raw_value=90.0,
            games_recent=10,
            rank=1,
        )
    )
    session.add(
        PlayerValue(
            player_name="Murphy, Marc",
            team="Carlton",
            as_of_season=2026,
            value=85.0,
            raw_value=85.0,
            games_sample=200,
        )
    )
    session.commit()

    squad = team_squad_rows(session, "Carlton", 2026, limit=5)
    assert [p["player_name"] for p in squad] == ["Walsh, Sam"]
