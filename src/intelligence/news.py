"""AFL news feed ingestion and injury headline parsing."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import feedparser
import requests

from src.config import SQUIGGLE_USER_AGENT

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "confident", "dominant", "strong", "return", "fit", "boost", "win", "star",
    "excellent", "momentum", "unstoppable", "premiership", "form",
}
NEGATIVE_WORDS = {
    "injury", "doubt", "crisis", "slump", "pressure", "sack", "loss", "concern",
    "struggle", "out", "suspended", "miss", "disappointing", "woeful",
}

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


def _score_text(text: str) -> tuple[float, float, float]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg + 1
    optimism = (pos - neg) / total
    pressure = neg / total
    stability = 1.0 - pressure
    return stability, pressure, optimism

RSS_FEEDS = [
    ("abc_afl", "https://www.abc.net.au/news/feed/7077144/rss.xml"),
    ("abc_sport", "https://www.abc.net.au/news/feed/2942460/rss.xml"),
]

AFL_HINTS = {"afl", "football", "premiership", "grand final", "brownlow"}

INJURY_KEYWORDS = {
    "injury",
    "injured",
    "ruled out",
    "out for",
    "will miss",
    "doubtful",
    "test",
    "scan",
    "sidelined",
    "concussion",
    "hamstring",
    "shoulder",
    "knee",
    "calf",
    "groin",
    "suspended",
    "omitted",
}

STATUS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("out", re.compile(r"\b(ruled out|will miss|out for|sidelined|omitted)\b", re.I)),
    ("doubtful", re.compile(r"\b(doubtful|test|scan|race against time)\b", re.I)),
    ("return", re.compile(r"\b(returns?|back in|cleared|available)\b", re.I)),
]

PLAYER_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z'-]+){0,2})\b"
)


@dataclass
class NewsArticle:
    title: str
    summary: str
    url: str
    published: str | None
    teams: list[str]
    sentiment: float
    is_injury: bool
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InjuryUpdate:
    player: str
    team: str
    status: str
    headline: str
    url: str
    published: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _match_teams(text: str) -> list[str]:
    lower = text.lower()
    matched: list[str] = []
    for team, keywords in TEAM_KEYWORDS.items():
        if any(kw.lower() in lower for kw in keywords):
            matched.append(team)
    return matched


def _article_tags(text: str) -> list[str]:
    lower = text.lower()
    tags: list[str] = []
    if any(k in lower for k in INJURY_KEYWORDS):
        tags.append("injury")
    if any(k in lower for k in ("lineup", "selected", "named", "teams")):
        tags.append("teams")
    if any(k in lower for k in ("trade", "contract", "draft")):
        tags.append("list")
    if any(k in lower for k in ("preview", "match", "clash", "face")):
        tags.append("preview")
    return tags


def _guess_player(title: str) -> str | None:
    """Best-effort player name from headline (AFL names are Title Case)."""
    skip = {
        "AFL", "The", "And", "For", "With", "After", "Before", "Round",
        "Match", "Team", "Coach", "Star", "Key", "Big", "New", "All",
    }
    for match in PLAYER_PATTERN.finditer(title):
        name = match.group(1).strip()
        parts = name.split()
        if parts[0] in skip:
            continue
        if len(parts) >= 2 or (len(parts) == 1 and len(parts[0]) > 4):
            return name
    return None


def _injury_status(text: str) -> str:
    for status, pattern in STATUS_PATTERNS:
        if pattern.search(text):
            return status
    return "monitoring"


def _fetch_feed(url: str) -> feedparser.FeedParserDict:
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": SQUIGGLE_USER_AGENT},
        )
        response.raise_for_status()
        return feedparser.parse(response.text)
    except Exception as exc:
        logger.warning("RSS fetch failed %s: %s", url, exc)
        return feedparser.FeedParserDict({})


def fetch_news_feed(*, limit: int = 60) -> list[NewsArticle]:
    articles: list[NewsArticle] = []
    seen: set[str] = set()

    for source, url in RSS_FEEDS:
        feed = _fetch_feed(url)
        per_source = limit // len(RSS_FEEDS) + 10

        for entry in feed.entries[:per_source]:
            title = entry.get("title", "").strip()
            if not title or title in seen:
                continue

            summary = entry.get("summary", entry.get("description", ""))
            summary = re.sub(r"<[^>]+>", " ", summary).strip()
            text = f"{title} {summary}"
            teams = _match_teams(text)

            # Sport feed: keep AFL-relevant stories only.
            if source == "abc_sport":
                lower = text.lower()
                if not teams and not any(h in lower for h in AFL_HINTS):
                    continue

            seen.add(title)
            _, _, optimism = _score_text(text)
            tags = _article_tags(text)
            is_injury = "injury" in tags

            published = None
            if entry.get("published_parsed"):
                try:
                    published = datetime(*entry.published_parsed[:6]).isoformat()
                except (TypeError, ValueError):
                    published = entry.get("published")
            else:
                published = entry.get("published")

            articles.append(
                NewsArticle(
                    title=title,
                    summary=summary[:280] + ("…" if len(summary) > 280 else ""),
                    url=entry.get("link", ""),
                    published=published,
                    teams=teams,
                    sentiment=float(max(-1, min(1, optimism))),
                    is_injury=is_injury,
                    tags=tags,
                )
            )

    articles.sort(
        key=lambda a: a.published or "",
        reverse=True,
    )
    return articles


def filter_news_for_teams(
    articles: list[NewsArticle],
    teams: list[str],
    *,
    limit: int = 12,
) -> list[NewsArticle]:
    team_set = set(teams)
    matched = [a for a in articles if team_set.intersection(a.teams)]
    if len(matched) < limit // 2:
        general = [a for a in articles if a.is_injury or "preview" in a.tags]
        for article in general:
            if article not in matched:
                matched.append(article)
            if len(matched) >= limit:
                break
    return matched[:limit]


def extract_injuries(
    articles: list[NewsArticle],
    teams: list[str],
    *,
    limit: int = 10,
) -> list[InjuryUpdate]:
    team_set = set(teams)
    injuries: list[InjuryUpdate] = []
    seen: set[tuple[str, str]] = set()

    for article in articles:
        if not article.is_injury:
            continue
        article_teams = [t for t in article.teams if t in team_set] or article.teams
        if team_set and not team_set.intersection(article.teams):
            continue

        player = _guess_player(article.title) or "Squad"
        status = _injury_status(article.title + " " + article.summary)
        for team in article_teams[:1] or teams[:1]:
            key = (player.lower(), team)
            if key in seen:
                continue
            seen.add(key)
            injuries.append(
                InjuryUpdate(
                    player=player,
                    team=team,
                    status=status,
                    headline=article.title,
                    url=article.url,
                    published=article.published,
                )
            )
            if len(injuries) >= limit:
                return injuries
    return injuries


def team_sentiments_from_articles(
    articles: list[NewsArticle],
    teams: list[str],
) -> dict[str, float]:
    scores: dict[str, list[float]] = {team: [] for team in teams}
    for article in articles:
        for team in article.teams:
            if team in scores:
                scores[team].append(article.sentiment)

    return {
        team: float(sum(vals) / len(vals)) if vals else 0.0
        for team, vals in scores.items()
    }
