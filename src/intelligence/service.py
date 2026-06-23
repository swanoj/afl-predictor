"""Orchestration for match intelligence with TTL caching."""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Match, StoredPrediction
from src.intelligence.briefing import generate_ai_briefing
from src.intelligence.market import get_market_edge
from src.intelligence.news import (
    NewsArticle,
    extract_injuries,
    fetch_news_feed,
    filter_news_for_teams,
    team_sentiments_from_articles,
)

_CACHE_TTL_SECONDS = 600
_feed_cache: dict[str, Any] = {"articles": [], "fetched_at": 0.0}


def _cached_feed() -> list[NewsArticle]:
    now = time.time()
    if now - _feed_cache["fetched_at"] > _CACHE_TTL_SECONDS:
        _feed_cache["articles"] = fetch_news_feed(limit=80)
        _feed_cache["fetched_at"] = now
    return _feed_cache["articles"]


def get_match_intelligence(
    session: Session,
    match: Match,
    *,
    stored: StoredPrediction | None = None,
) -> dict[str, Any]:
    """News, injuries, sentiment, and AI briefing for one match."""
    if stored is None:
        stored = session.scalars(
            select(StoredPrediction).where(StoredPrediction.match_id == match.id)
        ).first()

    home_wp = stored.home_win_prob if stored else 50.0
    away_wp = stored.away_win_prob if stored else 50.0
    predicted_winner = (
        stored.predicted_winner
        if stored
        else (match.home_team if home_wp >= away_wp else match.away_team)
    )
    margin = stored.predicted_margin if stored else 0.0

    teams = [match.home_team, match.away_team]
    feed = _cached_feed()
    news = filter_news_for_teams(feed, teams, limit=10)
    injuries = extract_injuries(feed, teams, limit=8)
    sentiments = team_sentiments_from_articles(feed, teams)

    briefing = generate_ai_briefing(
        home_team=match.home_team,
        away_team=match.away_team,
        venue=match.venue,
        home_win_prob=home_wp,
        away_win_prob=away_wp,
        predicted_margin=margin,
        predicted_winner=predicted_winner,
        home_sentiment=sentiments.get(match.home_team, 0.0),
        away_sentiment=sentiments.get(match.away_team, 0.0),
        news=news,
        injuries=injuries,
    )

    return {
        "match_id": match.id,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "venue": match.venue,
        "date": match.date.isoformat() if match.date else None,
        "news": [article.to_dict() for article in news],
        "injuries": [item.to_dict() for item in injuries],
        "sentiment": {
            match.home_team: sentiments.get(match.home_team, 0.0),
            match.away_team: sentiments.get(match.away_team, 0.0),
        },
        "briefing": briefing,
        "market_edge": get_market_edge(session, match, stored=stored),
        "feed_refreshed_at": _feed_cache["fetched_at"],
    }


def get_round_intelligence(
    session: Session,
    year: int,
    round_num: int,
) -> dict[str, Any]:
    """Round-level news feed and per-match injury counts."""
    matches = session.scalars(
        select(Match)
        .where(Match.year == year, Match.round == round_num)
        .order_by(Match.date, Match.id)
    ).all()

    teams = sorted({m.home_team for m in matches} | {m.away_team for m in matches})
    feed = _cached_feed()
    round_news = filter_news_for_teams(feed, teams, limit=20)
    all_injuries = extract_injuries(feed, teams, limit=30)

    injury_by_team: dict[str, int] = {team: 0 for team in teams}
    for injury in all_injuries:
        injury_by_team[injury.team] = injury_by_team.get(injury.team, 0) + 1

    match_summaries = []
    for match in matches:
        match_teams = [match.home_team, match.away_team]
        match_injuries = [
            i for i in all_injuries if i.team in match_teams
        ]
        match_summaries.append(
            {
                "match_id": match.id,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "injury_count": len(match_injuries),
                "news_count": len(
                    [a for a in round_news if set(a.teams) & set(match_teams)]
                ),
            }
        )

    return {
        "year": year,
        "round": round_num,
        "news": [article.to_dict() for article in round_news],
        "injuries": [item.to_dict() for item in all_injuries[:15]],
        "injury_by_team": injury_by_team,
        "matches": match_summaries,
        "feed_refreshed_at": _feed_cache["fetched_at"],
    }
