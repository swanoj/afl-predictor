"""Tests for similar-games RAG retrieval."""

import pytest
from sqlalchemy import select

from src.db.models import Match
from src.db.session import SessionLocal, init_db
from src.intelligence.similar_games import find_similar_games


def _completed_match(session, *, min_round: int = 10):
    return session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year >= 2020, Match.round >= min_round)  # noqa: E712
        .order_by(Match.year, Match.round)
    ).first()


def test_find_similar_games_shape():
    init_db()
    session = SessionLocal()
    try:
        match = _completed_match(session)
        if match is None:
            pytest.skip("No completed matches in DB")

        result = find_similar_games(session, match, limit=5)

        assert result["match_id"] == match.id
        assert "query" in result
        assert "elo_diff" in result["query"]
        assert "elo_bucket" in result["query"]
        assert "form_sign" in result["query"]
        assert isinstance(result["similar_games"], list)
        assert result["n_candidates"] >= len(result["similar_games"])
        assert len(result["similar_games"]) <= 5

        if result["similar_games"]:
            item = result["similar_games"][0]
            assert "match_id" in item
            assert "summary" in item
            assert "similarity_score" in item
            assert "home_win" in item
            assert item["match_id"] != match.id
    finally:
        session.close()


def test_similar_games_only_prior_completed():
    init_db()
    session = SessionLocal()
    try:
        match = _completed_match(session, min_round=15)
        if match is None:
            pytest.skip("No completed matches in DB")

        result = find_similar_games(session, match)
        for item in result["similar_games"]:
            assert (item["year"], item["round"]) <= (match.year, match.round)
            assert item["home_score"] is not None
    finally:
        session.close()
