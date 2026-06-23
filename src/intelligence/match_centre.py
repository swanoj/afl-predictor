"""Aggregated match-centre payload for the broadcast-style UI (numpy-free)."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Match, StoredPrediction
from src.intelligence.lineups import derive_match_lineups
from src.intelligence.market import get_market_edge
from src.intelligence.matchups import build_match_performance_layer
from src.intelligence.news import extract_injuries, fetch_news_feed, filter_news_for_teams
from src.intelligence.service import _cached_feed
from src.intelligence.teams import (
    compute_ladder,
    team_form,
    team_meta,
    team_season_summary,
    team_squad,
)
from src.predict.conformal import get_conformal_interval
from src.predict.serialize import stored_to_item


def _injuries_for_match(session: Session, match: Match) -> list:
    try:
        feed = _cached_feed()
    except Exception:
        feed = fetch_news_feed(limit=40)
    teams = [match.home_team, match.away_team]
    scoped = filter_news_for_teams(feed, teams, limit=15)
    return extract_injuries(scoped or feed, teams, limit=12)


def _team_centre_summary(
    session: Session,
    team: str,
    year: int,
    before_round: int,
    ladder: list[dict[str, Any]],
) -> dict[str, Any]:
    position = next((row["position"] for row in ladder if row["team"] == team), None)
    return {
        "meta": team_meta(team),
        "season": team_season_summary(session, team, year, before_round),
        "form": team_form(session, team, year, before_round),
        "ladder_position": position,
        "top_squad": team_squad(session, team, year, limit=5),
    }


def get_match_centre(session: Session, match_id: int) -> dict[str, Any]:
    """Full match-centre payload for one fixture."""
    match = session.get(Match, match_id)
    if match is None:
        raise ValueError("Match not found")

    stored = session.scalars(
        select(StoredPrediction).where(StoredPrediction.match_id == match_id)
    ).first()

    injuries = _injuries_for_match(session, match)
    lineups = derive_match_lineups(session, match, injuries)

    prediction: dict[str, Any] | None = None
    if stored is not None:
        prediction = stored_to_item(stored, match)
        prediction["model_version"] = stored.model_version
        prediction["win_prob_source"] = stored.win_prob_source
        prediction["margin_source"] = stored.margin_source

    player_projections: dict[str, Any] | None = None
    if stored is not None and stored.detail_json:
        try:
            detail = json.loads(stored.detail_json)
            player_projections = detail.get("player_projections")
        except json.JSONDecodeError:
            player_projections = None

    conformal: dict[str, Any] | None = None
    if stored is not None:
        interval = get_conformal_interval(stored.home_win_prob, match.year)
        conformal = {
            "home_win_prob": stored.home_win_prob,
            "year": match.year,
            **interval,
        }

    ladder = compute_ladder(session, match.year)
    before_round = match.round

    performance = build_match_performance_layer(
        session,
        match,
        lineups,
        player_projections=player_projections,
    )

    return {
        "match_id": match.id,
        "fixture": {
            "year": match.year,
            "round": match.round,
            "date": match.date.isoformat() if match.date else None,
            "venue": match.venue,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "complete": bool(match.complete),
        },
        "prediction": prediction,
        "lineups": lineups,
        "injuries": [item.to_dict() for item in injuries],
        "player_projections": player_projections,
        "performance": performance,
        "home": _team_centre_summary(
            session, match.home_team, match.year, before_round, ladder
        ),
        "away": _team_centre_summary(
            session, match.away_team, match.year, before_round, ladder
        ),
        "ladder": ladder,
        "conformal": conformal,
        "market_edge": get_market_edge(session, match, stored=stored),
    }
