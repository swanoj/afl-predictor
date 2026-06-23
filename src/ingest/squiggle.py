"""Squiggle API ingestion."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import SQUIGGLE_BASE_URL, SQUIGGLE_USER_AGENT

logger = logging.getLogger(__name__)


def fetch(query: str, **params: Any) -> list[dict]:
    """Fetch data from Squiggle API."""
    api_params = {"q": query, **params}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
    response = requests.get(
        SQUIGGLE_BASE_URL, params=api_params, headers=headers, timeout=60
    )
    response.raise_for_status()
    data = response.json()
    if query not in data:
        return []
    return data[query]


def fetch_games(year: int, round_num: int | None = None) -> list[dict]:
    params: dict[str, Any] = {"year": year}
    if round_num is not None:
        params["round"] = round_num
    return fetch("games", **params)


def fetch_teams() -> list[dict]:
    return fetch("teams")


def fetch_standings(year: int) -> list[dict]:
    return fetch("standings", year=year)


def fetch_tips(year: int, round_num: int | None = None) -> list[dict]:
    params: dict[str, Any] = {"year": year}
    if round_num is not None:
        params["round"] = round_num
    return fetch("tips", **params)


def parse_game_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def store_tips(session: Session, raw_tips: list[dict]) -> int:
    """Upsert raw Squiggle tip dicts into the ``squiggle_tips`` table.

    ``hconfidence`` (a 0-100 percentage) is normalised to a 0-1 home-win
    probability. Returns the number of rows written/updated.
    """
    from src.db.models import SquiggleTip

    written = 0
    for t in raw_tips:
        gameid = t.get("gameid")
        sourceid = t.get("sourceid")
        if gameid is None or sourceid is None:
            continue
        hconf = _to_float(t.get("hconfidence"))
        home_win_prob = hconf / 100.0 if hconf is not None else None

        existing = session.scalars(
            select(SquiggleTip).where(
                SquiggleTip.gameid == int(gameid),
                SquiggleTip.sourceid == int(sourceid),
            )
        ).first()
        target = existing or SquiggleTip(gameid=int(gameid), sourceid=int(sourceid))
        target.year = int(t.get("year")) if t.get("year") is not None else 0
        target.round = int(t.get("round")) if t.get("round") is not None else 0
        target.source = str(t.get("source", ""))
        target.home_team = t.get("hteam")
        target.away_team = t.get("ateam")
        target.tip = t.get("tip")
        target.home_win_prob = home_win_prob
        target.confidence = _to_float(t.get("confidence"))
        target.predicted_margin = _to_float(t.get("hmargin"))
        correct = t.get("correct")
        target.correct = int(correct) if correct is not None else None
        target.updated = t.get("updated")
        if existing is None:
            session.add(target)
        written += 1
    session.flush()
    return written


def load_tips_from_db(session: Session, year: int) -> list[dict]:
    """Load cached tips for a season from the DB as plain dicts."""
    from src.db.models import SquiggleTip

    rows = session.scalars(
        select(SquiggleTip).where(SquiggleTip.year == year)
    ).all()
    return [
        {
            "gameid": r.gameid,
            "year": r.year,
            "round": r.round,
            "sourceid": r.sourceid,
            "source": r.source,
            "hteam": r.home_team,
            "ateam": r.away_team,
            "tip": r.tip,
            "home_win_prob": r.home_win_prob,
            "confidence": r.confidence,
            "hmargin": r.predicted_margin,
            "correct": r.correct,
        }
        for r in rows
    ]


def get_season_tips(
    session: Session,
    year: int,
    use_cache: bool = True,
    persist: bool = True,
    sleep_s: float = 1.0,
) -> list[dict]:
    """Return all tips for a season, preferring the DB cache.

    Falls back to the Squiggle API (one polite request for the whole year)
    when the cache is empty, optionally persisting the result. Each returned
    dict has a normalised ``home_win_prob`` key in [0, 1].
    """
    if use_cache:
        cached = load_tips_from_db(session, year)
        if cached:
            logger.info("Loaded %d cached tips for %s", len(cached), year)
            return cached

    logger.info("Fetching tips for %s from Squiggle API", year)
    raw = fetch_tips(year)
    time.sleep(sleep_s)
    if not raw:
        return []
    if persist:
        store_tips(session, raw)
        session.commit()
    # Normalise to the same dict shape returned by load_tips_from_db.
    return load_tips_from_db(session, year) if persist else [
        {
            **t,
            "home_win_prob": (
                _to_float(t.get("hconfidence")) / 100.0
                if _to_float(t.get("hconfidence")) is not None
                else None
            ),
        }
        for t in raw
    ]
