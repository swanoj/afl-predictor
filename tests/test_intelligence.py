"""Tests for intelligence news parsing, briefings, and market edge."""

from src.intelligence.briefing import generate_rule_briefing
from src.intelligence.market import (
    MarketSnapshot,
    compare_model_to_market,
    _devig_two_way,
    _kelly_fraction,
    _mock_snapshot,
    _parse_odds_event,
)
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


def test_origin_headline_not_carlton():
    """NSW Blues Origin rugby must not match Carlton AFL."""
    from src.intelligence.news import _match_teams

    title = "Blues dealt fresh blow with Mitchell ruled out of Origin III"
    assert _match_teams(title) == []


def test_carlton_afl_headline_matches():
    from src.intelligence.news import _match_teams

    assert "Carlton" in _match_teams("Carlton midfielder ruled out with hamstring injury")


def test_devig_two_way():
    home, away = _devig_two_way(0.55, 0.55)
    assert abs(home - 0.5) < 1e-9
    assert abs(away - 0.5) < 1e-9


def test_kelly_fraction_positive_edge():
    # model 60%, odds 2.0 -> full Kelly = (0.6*2 - 1)/1 = 0.2
    assert abs(_kelly_fraction(0.6, 2.0) - 0.2) < 1e-9


def test_kelly_fraction_no_edge():
    assert _kelly_fraction(0.4, 2.0) == 0.0


def test_compare_model_to_market_home_edge():
    market = MarketSnapshot(
        source="mock",
        home_implied_prob=50.0,
        away_implied_prob=50.0,
        home_decimal_odds=2.0,
        away_decimal_odds=2.0,
    )
    result = compare_model_to_market(
        58.0,
        market,
        match_id=1,
        home_team="Carlton",
        away_team="Collingwood",
    )
    assert result.bet_side == "home"
    assert result.edge_pct == 8.0
    assert result.kelly_fraction > 0
    assert "Carlton" in result.recommendation


def test_compare_model_to_market_pass():
    market = MarketSnapshot(
        source="mock",
        home_implied_prob=52.0,
        away_implied_prob=48.0,
        home_decimal_odds=1.92,
        away_decimal_odds=2.08,
    )
    result = compare_model_to_market(53.0, market)
    assert result.bet_side == "none"
    assert "Pass" in result.recommendation


def test_parse_odds_event():
    event = {
        "home_team": "Carlton Blues",
        "away_team": "Collingwood Magpies",
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Carlton Blues", "price": 1.90},
                            {"name": "Collingwood Magpies", "price": 1.95},
                        ],
                    }
                ]
            }
        ],
    }
    parsed = _parse_odds_event(event)
    assert parsed is not None
    home_p, away_p, home_odds, away_odds, n = parsed
    assert abs(home_p + away_p - 1.0) < 1e-9
    assert n == 1


def test_mock_snapshot_deterministic():
    from types import SimpleNamespace

    match = SimpleNamespace(id=42, home_team="Carlton", away_team="Collingwood")
    a = _mock_snapshot(match, 55.0)
    b = _mock_snapshot(match, 55.0)
    assert a.home_implied_prob == b.home_implied_prob
    assert a.source == "mock"
