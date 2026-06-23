"""Find similar historical AFL matches for RAG-style context.

Uses only the ``matches`` table (no player logs / team stats) so it works on
the lean serving DB. Feature matching is intentionally simple and numpy-free:

* ``elo_diff`` bucket (50-point bands)
* same normalised venue
* ``form_diff`` sign (home momentum minus away momentum)
* same ``home_team`` identity

Historical candidates must be completed and strictly before the query match.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import ELO_HGA, ELO_K, ELO_START
from src.db.models import Match

# ---------------------------------------------------------------------------
# Lightweight walk-forward Elo + form (matches table only)
# ---------------------------------------------------------------------------

ELO_BUCKET_WIDTH = 50
FORM_WINDOW = 5


def _expected_score(r_home_adj: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_away - r_home_adj) / 400.0))


def _margin_multiplier(margin: int, elo_diff: float) -> float:
    return math.log(abs(margin) + 1) * (2.2 / (0.001 * abs(elo_diff) + 2.2))


def _update_elo(
    r_home: float, r_away: float, margin: int
) -> tuple[float, float]:
    r_home_adj = r_home + ELO_HGA
    elo_diff = r_home_adj - r_away
    exp_home = _expected_score(r_home_adj, r_away)
    if margin > 0:
        actual_home = 1.0
    elif margin < 0:
        actual_home = 0.0
    else:
        actual_home = 0.5
    mov = _margin_multiplier(margin, elo_diff)
    delta = ELO_K * mov * (actual_home - exp_home)
    return r_home + delta, r_away - delta


def _normalize_venue(venue: str | None) -> str:
    if not venue:
        return ""
    return re.sub(r"\s+", " ", venue.strip().lower())


def _elo_bucket(elo_diff: float) -> int:
    return int(math.floor(elo_diff / ELO_BUCKET_WIDTH))


def _form_sign(form_diff: float) -> str:
    if form_diff > 0.05:
        return "positive"
    if form_diff < -0.05:
        return "negative"
    return "neutral"


def _win_streak(margins: list[float]) -> int:
    streak = 0
    for margin in reversed(margins):
        if margin > 0:
            streak += 1
        else:
            break
    return streak


def _momentum(recent_margins: list[float], win_streak: int) -> float:
    if not recent_margins:
        return 0.0
    avg = sum(recent_margins) / len(recent_margins)
    raw = avg / 40.0 + win_streak * 0.1
    return max(-1.0, min(1.0, raw))


@dataclass
class _MatchFeatures:
    match_id: int
    year: int
    round: int
    home_team: str
    away_team: str
    venue: str | None
    elo_diff: float
    form_diff: float
    home_score: int | None = None
    away_score: int | None = None

    @property
    def venue_norm(self) -> str:
        return _normalize_venue(self.venue)

    @property
    def elo_bucket(self) -> int:
        return _elo_bucket(self.elo_diff)

    @property
    def form_sign(self) -> str:
        return _form_sign(self.form_diff)


def _team_margins_before(
    team: str,
    ordered_matches: list[Match],
    cutoff_index: int,
) -> list[float]:
    margins: list[float] = []
    for m in ordered_matches[:cutoff_index]:
        if m.home_score is None or m.away_score is None:
            continue
        if m.home_team == team:
            margins.append(float(m.home_score - m.away_score))
        elif m.away_team == team:
            margins.append(float(m.away_score - m.home_score))
    return margins


def _features_at_index(
    ordered_matches: list[Match], index: int, ratings: dict[str, float]
) -> _MatchFeatures:
    m = ordered_matches[index]
    home_margins = _team_margins_before(m.home_team, ordered_matches, index)
    away_margins = _team_margins_before(m.away_team, ordered_matches, index)

    home_recent = home_margins[-FORM_WINDOW:]
    away_recent = away_margins[-FORM_WINDOW:]
    home_form = _momentum(home_recent, _win_streak(home_margins))
    away_form = _momentum(away_recent, _win_streak(away_margins))

    r_home = ratings.get(m.home_team, ELO_START)
    r_away = ratings.get(m.away_team, ELO_START)
    elo_diff = (r_home + ELO_HGA) - r_away

    return _MatchFeatures(
        match_id=m.id,
        year=m.year,
        round=m.round,
        home_team=m.home_team,
        away_team=m.away_team,
        venue=m.venue,
        elo_diff=elo_diff,
        form_diff=home_form - away_form,
        home_score=m.home_score,
        away_score=m.away_score,
    )


def _similarity_score(query: _MatchFeatures, candidate: _MatchFeatures) -> float:
    score = 0.0
    if candidate.home_team == query.home_team:
        score += 3.0
    if candidate.venue_norm and candidate.venue_norm == query.venue_norm:
        score += 2.0
    if candidate.elo_bucket == query.elo_bucket:
        score += 2.0
    elif abs(candidate.elo_bucket - query.elo_bucket) == 1:
        score += 1.0
    if candidate.form_sign == query.form_sign:
        score += 1.0
    # Prefer closer elo_diff within the same matchup shape.
    score -= abs(candidate.elo_diff - query.elo_diff) / 200.0
    return score


def _outcome_summary(candidate: _MatchFeatures) -> str:
    if candidate.home_score is None or candidate.away_score is None:
        return "Result unavailable"
    margin = candidate.home_score - candidate.away_score
    if margin > 0:
        winner = candidate.home_team
    elif margin < 0:
        winner = candidate.away_team
        margin = -margin
    else:
        return (
            f"Draw {candidate.home_team} {candidate.home_score}-"
            f"{candidate.away_score} {candidate.away_team} "
            f"(R{candidate.round} {candidate.year})"
        )
    venue_bit = f" at {candidate.venue}" if candidate.venue else ""
    return (
        f"{winner} beat "
        f"{candidate.away_team if winner == candidate.home_team else candidate.home_team} "
        f"by {margin}{venue_bit} (R{candidate.round} {candidate.year})"
    )


def _brief_summary(query: _MatchFeatures, candidate: _MatchFeatures) -> str:
    bits: list[str] = []
    if candidate.home_team == query.home_team:
        bits.append(f"same home side ({query.home_team})")
    if candidate.venue_norm and candidate.venue_norm == query.venue_norm:
        bits.append(f"same venue ({candidate.venue})")
    if candidate.elo_bucket == query.elo_bucket:
        bits.append("similar Elo edge")
    if candidate.form_sign == query.form_sign and query.form_sign != "neutral":
        bits.append(f"home form {query.form_sign}")
    context = ", ".join(bits) if bits else " comparable matchup profile"
    return f"{_outcome_summary(candidate)};{context}."


def find_similar_games(
    session: Session,
    match: Match,
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Return the top *limit* completed historical matches similar to *match*."""
    ordered = session.scalars(
        select(Match).order_by(Match.year, Match.round, Match.id)
    ).all()

    index_by_id = {m.id: i for i, m in enumerate(ordered)}
    query_index = index_by_id.get(match.id)
    if query_index is None:
        raise ValueError(f"Match {match.id} not found in matches table")

    ratings: dict[str, float] = {}
    historical: list[_MatchFeatures] = []

    for i, m in enumerate(ordered):
        if i >= query_index:
            break
        if not m.complete or m.home_score is None or m.away_score is None:
            continue
        feat = _features_at_index(ordered, i, ratings)
        feat.home_score = m.home_score
        feat.away_score = m.away_score
        historical.append(feat)
        ratings[m.home_team] = ratings.get(m.home_team, ELO_START)
        ratings[m.away_team] = ratings.get(m.away_team, ELO_START)
        r_home, r_away = _update_elo(
            ratings[m.home_team],
            ratings[m.away_team],
            m.home_score - m.away_score,
        )
        ratings[m.home_team] = r_home
        ratings[m.away_team] = r_away

    query = _features_at_index(ordered, query_index, ratings)

    scored: list[tuple[float, _MatchFeatures]] = [
        (_similarity_score(query, cand), cand) for cand in historical
    ]
    scored.sort(key=lambda t: (-t[0], -t[1].year, -t[1].round))

    similar: list[dict[str, Any]] = []
    for sim_score, cand in scored[:limit]:
        margin = (cand.home_score or 0) - (cand.away_score or 0)
        similar.append(
            {
                "match_id": cand.match_id,
                "year": cand.year,
                "round": cand.round,
                "home_team": cand.home_team,
                "away_team": cand.away_team,
                "venue": cand.venue,
                "home_score": cand.home_score,
                "away_score": cand.away_score,
                "winner": (
                    cand.home_team
                    if margin > 0
                    else cand.away_team
                    if margin < 0
                    else None
                ),
                "margin": margin,
                "home_win": margin > 0,
                "similarity_score": round(sim_score, 2),
                "summary": _brief_summary(query, cand),
            }
        )

    return {
        "match_id": match.id,
        "query": {
            "home_team": query.home_team,
            "away_team": query.away_team,
            "venue": query.venue,
            "elo_diff": round(query.elo_diff, 1),
            "elo_bucket": query.elo_bucket,
            "form_diff": round(query.form_diff, 3),
            "form_sign": query.form_sign,
        },
        "similar_games": similar,
        "n_candidates": len(historical),
    }
