"""Narrative sentiment ingestion (Phase 5 — optional)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "confident", "dominant", "strong", "return", "fit", "boost", "win", "star",
    "excellent", "momentum", "unstoppable", "premiership", "form",
}
NEGATIVE_WORDS = {
    "injury", "doubt", "crisis", "slump", "pressure", "sack", "loss", "concern",
    "struggle", "out", "suspended", "miss", "disappointing", "woeful",
}


@dataclass
class SentimentScore:
    team: str
    stability: float
    media_pressure: float
    fan_optimism: float
    composite: float
    scraped_at: datetime


def _score_text(text: str) -> tuple[float, float, float]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg + 1
    optimism = (pos - neg) / total
    pressure = neg / total
    stability = 1.0 - pressure
    return stability, pressure, optimism


def scrape_afl_rss(team_keywords: dict[str, list[str]]) -> list[SentimentScore]:
    """Scrape AFL.com.au RSS and score team mentions."""
    feeds = [
        "https://www.afl.com.au/news/rss.xml",
    ]
    scores: dict[str, list[tuple[float, float, float]]] = {t: [] for t in team_keywords}

    for url in feeds:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.warning("RSS fetch failed %s: %s", url, e)
            continue

        for entry in feed.entries[:50]:
            text = f"{entry.get('title', '')} {entry.get('summary', '')}"
            for team, keywords in team_keywords.items():
                if any(kw.lower() in text.lower() for kw in keywords):
                    scores[team].append(_score_text(text))

    results = []
    now = datetime.utcnow()
    for team, entries in scores.items():
        if not entries:
            results.append(
                SentimentScore(team, 0.5, 0.5, 0.0, 0.0, now)
            )
            continue
        stabilities = [e[0] for e in entries]
        pressures = [e[1] for e in entries]
        optimisms = [e[2] for e in entries]
        composite = float(sum(optimisms) / len(optimisms))
        results.append(
            SentimentScore(
                team=team,
                stability=float(sum(stabilities) / len(stabilities)),
                media_pressure=float(sum(pressures) / len(pressures)),
                fan_optimism=composite,
                composite=composite,
                scraped_at=now,
            )
        )
    return results


TEAM_KEYWORDS: dict[str, list[str]] = {
    "Adelaide": ["Adelaide", "Crows"],
    "Brisbane Lions": ["Brisbane", "Lions"],
    "Carlton": ["Carlton", "Blues"],
    "Collingwood": ["Collingwood", "Magpies"],
    "Essendon": ["Essendon", "Bombers"],
    "Fremantle": ["Fremantle", "Dockers"],
    "Geelong": ["Geelong", "Cats"],
    "Gold Coast": ["Gold Coast", "Suns"],
    "GWS": ["GWS", "Giants"],
    "Hawthorn": ["Hawthorn", "Hawks"],
    "Melbourne": ["Melbourne", "Demons"],
    "North Melbourne": ["North Melbourne", "Kangaroos"],
    "Port Adelaide": ["Port Adelaide", "Power"],
    "Richmond": ["Richmond", "Tigers"],
    "St Kilda": ["St Kilda", "Saints"],
    "Sydney": ["Sydney", "Swans"],
    "West Coast": ["West Coast", "Eagles"],
    "Western Bulldogs": ["Western Bulldogs", "Bulldogs", "Footscray"],
}


def get_team_sentiments() -> dict[str, float]:
    """Return composite sentiment per team (-1 to 1)."""
    scores = scrape_afl_rss(TEAM_KEYWORDS)
    return {s.team: float(max(-1, min(1, s.composite))) for s in scores}
