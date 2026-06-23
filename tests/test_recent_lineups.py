"""Tests for last-match lineup selection."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Match, PlayerGameLog, ServingRecentLineup
from src.intelligence.lineups import derive_team_lineup
from src.intelligence.recent_lineups import recent_match_lineup
from src.predict.lineup_adjust import PlayerValueMap


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def test_recent_match_lineup_from_logs(session):
    session.add(
        Match(
            id=1,
            squiggle_id=1,
            year=2025,
            round=24,
            home_team="Sydney",
            away_team="West Coast",
            home_score=90,
            away_score=60,
            complete=True,
        )
    )
    session.add_all(
        [
            PlayerGameLog(
                match_id=1,
                player_name="Gulden, Errol",
                team="Sydney",
                opponent="West Coast",
                year=2025,
                round=24,
                disposals=30,
            ),
            PlayerGameLog(
                match_id=1,
                player_name="Heeney, Isaac",
                team="Sydney",
                opponent="West Coast",
                year=2025,
                round=24,
                disposals=28,
            ),
            PlayerGameLog(
                match_id=1,
                player_name="Rampe, Dane",
                team="Sydney",
                opponent="West Coast",
                year=2025,
                round=24,
                disposals=12,
            ),
        ]
    )
    session.commit()

    result = recent_match_lineup(session, "Sydney", 2026, before_round=1)
    assert result is not None
    lineup, meta = result
    assert lineup == ["Gulden, Errol", "Heeney, Isaac", "Rampe, Dane"]
    assert meta["source_year"] == 2025
    assert meta["source_round"] == 24


def test_derive_team_lineup_uses_recent_match(session):
    session.add(
        Match(
            id=1,
            squiggle_id=1,
            year=2025,
            round=10,
            home_team="Sydney",
            away_team="Carlton",
            home_score=80,
            away_score=70,
            complete=True,
        )
    )
    session.add(
        PlayerGameLog(
            match_id=1,
            player_name="Papley, Tom",
            team="Sydney",
            opponent="Carlton",
            year=2025,
            round=10,
            disposals=20,
        )
    )
    session.add(
        ServingRecentLineup(
            team="Sydney",
            season=2026,
            player_name="Papley, Tom",
            slot=1,
            source_year=2025,
            source_round=10,
            source_opponent="Carlton",
        )
    )
    session.commit()

    values: PlayerValueMap = {("McInerney, Justin", "Sydney"): 99.0}
    lineup = derive_team_lineup(session, "Sydney", 2026, 5, [], values)
    assert lineup["expected_lineup"] == ["Papley, Tom"]
    assert lineup["lineup_source"] == "last_match"
    assert "Papley, Tom" in lineup["lineup_values"]
