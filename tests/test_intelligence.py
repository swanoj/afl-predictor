"""Tests for intelligence news parsing and briefings."""

from src.intelligence.briefing import generate_rule_briefing
from src.intelligence.news import (
    NewsArticle,
    extract_injuries,
    filter_news_for_teams,
    team_sentiments_from_articles,
)


def test_filter_news_for_teams():
    articles = [
        NewsArticle(
            title="Carlton star ruled out",
            summary="Hamstring injury",
            url="https://example.com/1",
            published="2026-06-01T00:00:00",
            teams=["Carlton"],
            sentiment=-0.4,
            is_injury=True,
            tags=["injury"],
        ),
        NewsArticle(
            title="AFL round preview",
            summary="General preview",
            url="https://example.com/2",
            published="2026-06-02T00:00:00",
            teams=[],
            sentiment=0.0,
            is_injury=False,
            tags=["preview"],
        ),
    ]
    filtered = filter_news_for_teams(articles, ["Carlton", "Collingwood"], limit=5)
    assert any(a.title.startswith("Carlton") for a in filtered)


def test_extract_injuries():
    articles = [
        NewsArticle(
            title="Marcus Bontempelli ruled out for Bulldogs clash",
            summary="Shoulder injury",
            url="https://example.com/3",
            published=None,
            teams=["Western Bulldogs"],
            sentiment=-0.5,
            is_injury=True,
            tags=["injury"],
        )
    ]
    injuries = extract_injuries(articles, ["Western Bulldogs"])
    assert len(injuries) >= 1
    assert injuries[0].status == "out"


def test_team_sentiments():
    articles = [
        NewsArticle(
            title="Richmond dominant win",
            summary="Excellent form",
            url="https://example.com/4",
            published=None,
            teams=["Richmond"],
            sentiment=0.6,
            is_injury=False,
            tags=[],
        )
    ]
    scores = team_sentiments_from_articles(articles, ["Richmond"])
    assert scores["Richmond"] > 0


def test_rule_briefing():
    briefing = generate_rule_briefing(
        home_team="Carlton",
        away_team="Collingwood",
        venue="MCG",
        home_win_prob=55.0,
        away_win_prob=45.0,
        predicted_margin=8.0,
        predicted_winner="Carlton",
        home_sentiment=0.1,
        away_sentiment=-0.1,
        news=[],
        injuries=[],
    )
    assert "Carlton" in briefing["headline"]
    assert briefing["source"] == "rules"
    assert len(briefing["key_factors"]) >= 2
