"""Tests for player performance and matchup layer."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    Base,
    Match,
    PlayerGameLog,
    ServingPlayerOpponentSplit,
    ServingPlayerProfile,
)
from src.intelligence.lineups import derive_team_lineup
from src.intelligence.matchups import build_match_performance_layer
from src.predict.lineup_adjust import PlayerValueMap


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _add_match(s, mid, year, rnd, home, away, hs, aw):
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
            complete=True,
        )
    )


def test_build_match_performance_layer(session):
    _add_match(session, 1, 2026, 1, "Carlton", "Richmond", 90, 70)
    _add_match(session, 2, 2026, 2, "Carlton", "Collingwood", 80, 85)
    _add_match(session, 3, 2026, 1, "Richmond", "Essendon", 75, 70)

    session.add_all(
        [
            PlayerGameLog(
                match_id=1,
                player_name="Walsh, Sam",
                team="Carlton",
                opponent="Richmond",
                year=2026,
                round=1,
                disposals=32,
                goals=1,
                tackles=5,
                clearances=8,
                hit_outs=0,
            ),
            PlayerGameLog(
                match_id=2,
                player_name="Walsh, Sam",
                team="Carlton",
                opponent="Collingwood",
                year=2026,
                round=2,
                disposals=28,
                goals=0,
                tackles=4,
                clearances=7,
                hit_outs=0,
            ),
            PlayerGameLog(
                match_id=1,
                player_name="Short, Tom",
                team="Richmond",
                opponent="Carlton",
                year=2026,
                round=1,
                disposals=22,
                goals=0,
                tackles=6,
                clearances=4,
                hit_outs=0,
            ),
        ]
    )

    session.add_all(
        [
            ServingPlayerProfile(
                team="Carlton",
                season=2026,
                player_name="Walsh, Sam",
                role="Inside Mid",
                disposals_avg=30.0,
                goals_avg=0.5,
                tackles_avg=4.5,
                clearances_avg=7.5,
                hit_outs_avg=0.0,
                games=2,
                form_disposals=30.0,
                form_goals=0.5,
                form_games=2,
            ),
            ServingPlayerProfile(
                team="Richmond",
                season=2026,
                player_name="Short, Tom",
                role="Defender",
                disposals_avg=22.0,
                goals_avg=0.0,
                tackles_avg=6.0,
                clearances_avg=4.0,
                hit_outs_avg=0.0,
                games=1,
                form_disposals=22.0,
                form_goals=0.0,
                form_games=1,
            ),
            ServingPlayerOpponentSplit(
                team="Carlton",
                season=2026,
                player_name="Walsh, Sam",
                opponent="Richmond",
                disposals_avg=32.0,
                goals_avg=1.0,
                games=1,
            ),
        ]
    )
    session.commit()

    match = session.get(Match, 2)
    values: PlayerValueMap = {
        ("Walsh, Sam", "Carlton"): 90.0,
        ("Short, Tom", "Richmond"): 70.0,
    }
    lineups = {
        "home": derive_team_lineup(
            session, "Carlton", 2026, 3, [], values
        ),
        "away": derive_team_lineup(
            session, "Collingwood", 2026, 3, [], values
        ),
    }

    layer = build_match_performance_layer(session, match, lineups)
    walsh = next(
        p for p in layer["home_players"] if p["player_name"] == "Walsh, Sam"
    )
    assert walsh["season_disposals"] == 30.0
    assert walsh["matchup_grade"] in {"A", "B", "C", "D", "–"}
    assert layer["team_profiles"]["home"].get("avg_scored", 0) > 0
    assert len(layer["key_matchups"]) >= 1
