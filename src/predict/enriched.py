"""Lightweight prediction enrichment for the serving API (no numpy)."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.intelligence.lineups import (
    derive_match_lineups,
    expected_lineup,
    load_player_values_map,
    roster_candidates_from_values,
    baseline_lineup_value_proxy,
    replacement_level,
)
from src.intelligence.news import extract_injuries, fetch_news_feed, filter_news_for_teams
from src.intelligence.service import _cached_feed
from src.predict.lineup_adjust import apply_lineup_adjustment, lineup_value
from src.predict.conformal import get_conformal_interval


def _injuries_for_match(session: Session, match) -> list:
    try:
        feed = _cached_feed()
    except Exception:
        feed = fetch_news_feed(limit=40)
    teams = [match.home_team, match.away_team]
    scoped = filter_news_for_teams(feed, teams, limit=15)
    return extract_injuries(scoped or feed, teams, limit=12)


def enrich_prediction_item(
    session: Session,
    match,
    item: dict[str, Any],
    *,
    injuries=None,
) -> dict[str, Any]:
    """Add lineup-adjusted win probs and conformal interval to a compact row."""
    out = dict(item)
    values = load_player_values_map(session, match.year)
    if not values:
        out["lineup_win_prob_shift"] = None
        return out

    if injuries is None:
        injuries = _injuries_for_match(session, match)

    lineups = derive_match_lineups(session, match, injuries)
    base_home = float(item["home_win_prob"])
    adjusted = apply_lineup_adjustment(
        base_home,
        lineups["home"]["expected_lineup"],
        lineups["away"]["expected_lineup"],
        match.home_team,
        match.away_team,
        lineups["home"]["baseline_lineup_value"],
        lineups["away"]["baseline_lineup_value"],
        values,
    )

    shift = round(float(adjusted["home_win_prob"]) - base_home, 2)
    out["lineup_win_prob_shift"] = shift
    out["lineup_adjusted_home_win_prob"] = round(float(adjusted["home_win_prob"]), 1)
    out["lineup_adjusted_away_win_prob"] = round(float(adjusted["away_win_prob"]), 1)
    out["lineup_margin_adj"] = round(float(adjusted["lineup_margin_adj"]), 2)
    out["home_out_players"] = lineups["home"]["out_players"]
    out["away_out_players"] = lineups["away"]["out_players"]

    interval = get_conformal_interval(base_home, match.year)
    out["conformal_lower"] = interval["lower"]
    out["conformal_upper"] = interval["upper"]
    out["conformal_coverage"] = interval["coverage"]
    return out


def run_whatif(
    session: Session,
    match,
    *,
    home_out: list[str],
    away_out: list[str],
    base_home_prob: float | None = None,
) -> dict[str, Any]:
    """Counterfactual win prob when toggling players OUT."""
    from sqlalchemy import select

    from src.db.models import StoredPrediction

    stored = session.scalars(
        select(StoredPrediction).where(StoredPrediction.match_id == match.id)
    ).first()
    if stored is None and base_home_prob is None:
        raise ValueError("No stored prediction for match")

    base_home = float(base_home_prob if base_home_prob is not None else stored.home_win_prob)
    values = load_player_values_map(session, match.year)
    rep = replacement_level(values)

    home_candidates = roster_candidates_from_values(values, match.home_team)
    away_candidates = roster_candidates_from_values(values, match.away_team)

    home_out_set = {p.lower() for p in home_out}
    away_out_set = {p.lower() for p in away_out}

    home_lineup = expected_lineup(
        [p for p in home_candidates if p.lower() not in home_out_set],
        set(),
    )
    away_lineup = expected_lineup(
        [p for p in away_candidates if p.lower() not in away_out_set],
        set(),
    )

    home_baseline = baseline_lineup_value_proxy(home_candidates, match.home_team, values, replacement=rep)
    away_baseline = baseline_lineup_value_proxy(away_candidates, match.away_team, values, replacement=rep)

    adjusted = apply_lineup_adjustment(
        base_home,
        home_lineup,
        away_lineup,
        match.home_team,
        match.away_team,
        home_baseline,
        away_baseline,
        values,
        replacement=rep,
    )

    return {
        "match_id": match.id,
        "base_home_win_prob": round(base_home, 1),
        "base_away_win_prob": round(100.0 - base_home, 1),
        "home_win_prob": round(float(adjusted["home_win_prob"]), 1),
        "away_win_prob": round(float(adjusted["away_win_prob"]), 1),
        "lineup_margin_adj": round(float(adjusted["lineup_margin_adj"]), 2),
        "home_lineup_value": round(lineup_value(home_lineup, match.home_team, values, rep), 1),
        "away_lineup_value": round(lineup_value(away_lineup, match.away_team, values, rep), 1),
        "home_out": home_out,
        "away_out": away_out,
        "win_prob_source": adjusted["win_prob_source"],
    }
