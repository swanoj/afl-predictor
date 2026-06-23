"""AI and rule-based match briefings."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.intelligence.news import InjuryUpdate, NewsArticle

logger = logging.getLogger(__name__)


def _format_news_bullets(articles: list[NewsArticle], limit: int = 4) -> str:
    lines = []
    for article in articles[:limit]:
        tone = "↑" if article.sentiment > 0.15 else "↓" if article.sentiment < -0.15 else "→"
        lines.append(f"- [{tone}] {article.title}")
    return "\n".join(lines) if lines else "- No major headlines in the last feed pull."


def _format_injuries(injuries: list[InjuryUpdate]) -> str:
    if not injuries:
        return "No flagged injury headlines for these sides."
    return "\n".join(
        f"- {item.player} ({item.team}): {item.status} — {item.headline}"
        for item in injuries[:6]
    )


def generate_rule_briefing(
    *,
    home_team: str,
    away_team: str,
    venue: str | None,
    home_win_prob: float,
    away_win_prob: float,
    predicted_margin: float,
    predicted_winner: str,
    home_sentiment: float,
    away_sentiment: float,
    news: list[NewsArticle],
    injuries: list[InjuryUpdate],
) -> dict[str, Any]:
    """Template briefing when no LLM key is configured."""
    fav = home_team if home_win_prob >= away_win_prob else away_team
    fav_prob = max(home_win_prob, away_win_prob)
    margin_abs = abs(predicted_margin)

    sentiment_note = []
    if home_sentiment > 0.2:
        sentiment_note.append(f"{home_team} media tone is upbeat")
    elif home_sentiment < -0.2:
        sentiment_note.append(f"{home_team} headlines skew negative")
    if away_sentiment > 0.2:
        sentiment_note.append(f"{away_team} media tone is upbeat")
    elif away_sentiment < -0.2:
        sentiment_note.append(f"{away_team} headlines skew negative")

    injury_note = ""
    outs = [i for i in injuries if i.status == "out"]
    doubtful = [i for i in injuries if i.status == "doubtful"]
    if outs:
        injury_note = (
            f"Injury wire flags {len(outs)} OUT headline(s) — "
            f"watch {', '.join(i.player for i in outs[:3])}."
        )
    elif doubtful:
        injury_note = (
            f"{len(doubtful)} player(s) listed doubtful in recent headlines."
        )

    headline = (
        f"{fav} tipped by {fav_prob:.0f}% model confidence"
        f"{' at ' + venue if venue else ''}"
    )

    summary = (
        f"The calibrated model makes {predicted_winner} a {fav_prob:.0f}% pick, "
        f"projecting a {margin_abs:.0f}-point margin. "
        + (" ".join(sentiment_note) + ". " if sentiment_note else "")
        + (injury_note + " " if injury_note else "")
        + "Headline win probability uses rolling form + Elo; news sentiment is "
        "informational only and does not override the model."
    )

    key_factors = [
        f"Model edge: {fav} ({fav_prob:.0f}%)",
        f"Predicted margin: {predicted_margin:+.0f} ({predicted_winner})",
    ]
    if venue:
        key_factors.append(f"Venue: {venue}")
    if injuries:
        key_factors.append(f"{len(injuries)} injury headline(s) in feed")
    if sentiment_note:
        key_factors.append(sentiment_note[0])

    watchlist = [
        article.title for article in news[:3]
    ] or ["Check team announcements 24h before bounce"]

    return {
        "source": "rules",
        "headline": headline,
        "summary": summary.strip(),
        "key_factors": key_factors,
        "injury_impact": injury_note or "No major injury flags in current headlines.",
        "news_watch": watchlist,
    }


def generate_ai_briefing(
    *,
    home_team: str,
    away_team: str,
    venue: str | None,
    home_win_prob: float,
    away_win_prob: float,
    predicted_margin: float,
    predicted_winner: str,
    home_sentiment: float,
    away_sentiment: float,
    news: list[NewsArticle],
    injuries: list[InjuryUpdate],
) -> dict[str, Any]:
    """OpenAI-powered briefing; falls back to rules if unavailable."""
    fallback = generate_rule_briefing(
        home_team=home_team,
        away_team=away_team,
        venue=venue,
        home_win_prob=home_win_prob,
        away_win_prob=away_win_prob,
        predicted_margin=predicted_margin,
        predicted_winner=predicted_winner,
        home_sentiment=home_sentiment,
        away_sentiment=away_sentiment,
        news=news,
        injuries=injuries,
    )

    if not OPENAI_API_KEY:
        return fallback

    prompt = f"""You are an AFL analyst writing a concise pre-match briefing for punters.

Match: {home_team} vs {away_team}
Venue: {venue or "TBC"}
Model win prob: {home_team} {home_win_prob:.1f}%, {away_team} {away_win_prob:.1f}%
Predicted winner: {predicted_winner} by {predicted_margin:+.0f}
Media sentiment ({home_team}): {home_sentiment:+.2f}
Media sentiment ({away_team}): {away_sentiment:+.2f}

Recent headlines:
{_format_news_bullets(news)}

Injury headlines:
{_format_injuries(injuries)}

Return JSON only with keys: headline (string, max 12 words), summary (2-3 sentences),
key_factors (array of 3-5 short strings), injury_impact (1 sentence),
news_watch (array of 2-3 strings). Be factual; don't invent player stats."""

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You write sharp AFL tipping briefs. JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.4,
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return {
                "source": "openai",
                "headline": parsed.get("headline", fallback["headline"]),
                "summary": parsed.get("summary", fallback["summary"]),
                "key_factors": parsed.get("key_factors", fallback["key_factors"]),
                "injury_impact": parsed.get("injury_impact", fallback["injury_impact"]),
                "news_watch": parsed.get("news_watch", fallback["news_watch"]),
            }
    except Exception as exc:
        logger.warning("AI briefing failed, using rules: %s", exc)
        fallback["source"] = "rules (AI unavailable)"
        return fallback
