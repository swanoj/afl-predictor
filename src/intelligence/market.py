"""Model vs market edge — compare our win probability to bookmaker / consensus odds.

Data sources (first match wins):
  1. The Odds API (``ODDS_API_KEY``) — live AFL h2h prices
  2. Squiggle tipster consensus from the DB cache (implied probabilities)
  3. Deterministic mock snapshot for demo / offline use

Lean-runtime safe: stdlib + requests/httpx + SQLAlchemy only (no pandas/numpy).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import ODDS_API_KEY, SQUIGGLE_USER_AGENT
from src.db.models import Match, SquiggleTip, StoredPrediction

logger = logging.getLogger(__name__)

MarketSource = Literal["odds_api", "squiggle_consensus", "mock"]

# Squiggle meta-aggregators — excluded from consensus so we don't double-count.
_META_SOURCES = frozenset({"Squiggle", "Aggregate"})

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
_ODDS_SPORT = "aussierules_afl"
_ODDS_REGIONS = "au"
_ODDS_MARKETS = "h2h"
_ODDS_CACHE_TTL = 600

_odds_cache: dict[str, Any] = {"events": [], "fetched_at": 0.0}

# Strip common bookmaker / Squiggle suffixes for fuzzy team matching.
_TEAM_SUFFIX_RE = re.compile(
    r"\s+(tigers|blues|magpies|bombers|dockers|cats|suns|giants|hawks|"
    r"demons|kangaroos|power|saints|swans|eagles|bulldogs|crows|lions|football club)\b",
    re.I,
)


@dataclass
class MarketSnapshot:
    """Normalised market view for one AFL match."""

    source: MarketSource
    home_implied_prob: float  # 0–100
    away_implied_prob: float  # 0–100
    home_decimal_odds: float | None = None
    away_decimal_odds: float | None = None
    n_bookmakers: int = 0
    n_tipsters: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketEdgeResult:
    """Model vs market comparison for a single match."""

    match_id: int
    home_team: str
    away_team: str
    model_home_win_prob: float
    model_away_win_prob: float
    market: MarketSnapshot
    edge_pct: float
    bet_side: Literal["home", "away", "none"]
    kelly_fraction: float
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "model_home_win_prob": round(self.model_home_win_prob, 2),
            "model_away_win_prob": round(self.model_away_win_prob, 2),
            "market_home_implied_prob": round(self.market.home_implied_prob, 2),
            "market_away_implied_prob": round(self.market.away_implied_prob, 2),
            "market_source": self.market.source,
            "home_decimal_odds": self.market.home_decimal_odds,
            "away_decimal_odds": self.market.away_decimal_odds,
            "edge_pct": round(self.edge_pct, 2),
            "bet_side": self.bet_side,
            "kelly_fraction": round(self.kelly_fraction, 4),
            "recommendation": self.recommendation,
            "n_bookmakers": self.market.n_bookmakers,
            "n_tipsters": self.market.n_tipsters,
            "market_note": self.market.note,
        }
        return payload


def _normalise_team(name: str) -> str:
    cleaned = _TEAM_SUFFIX_RE.sub("", name.strip().lower())
    return re.sub(r"[^a-z0-9]", "", cleaned)


def _teams_match(a: str, b: str) -> bool:
    na, nb = _normalise_team(a), _normalise_team(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _devig_two_way(home_raw: float, away_raw: float) -> tuple[float, float]:
    """Remove overround from implied probabilities (0–1 scale)."""
    total = home_raw + away_raw
    if total <= 0:
        return 0.5, 0.5
    return home_raw / total, away_raw / total


def _implied_from_decimal(odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    return 1.0 / odds


def _pct(prob_0_1: float) -> float:
    return prob_0_1 * 100.0


def _consensus_from_rows(rows: list[SquiggleTip]) -> MarketSnapshot | None:
    probs = [
        r.home_win_prob
        for r in rows
        if r.source not in _META_SOURCES and r.home_win_prob is not None
    ]
    if not probs:
        return None
    home_p = sum(probs) / len(probs)
    away_p = 1.0 - home_p
    return MarketSnapshot(
        source="squiggle_consensus",
        home_implied_prob=_pct(home_p),
        away_implied_prob=_pct(away_p),
        n_tipsters=len(probs),
        note=f"Squiggle consensus across {len(probs)} tipsters (vig-free normalisation N/A).",
    )


def _squiggle_consensus(session: Session, match: Match) -> MarketSnapshot | None:
    rows = session.scalars(
        select(SquiggleTip).where(
            SquiggleTip.gameid == match.squiggle_id,
            SquiggleTip.year == match.year,
        )
    ).all()
    return _consensus_from_rows(rows)


def _parse_odds_event(event: dict[str, Any]) -> tuple[float, float, float, float, int] | None:
    """Return (home_prob, away_prob, home_odds, away_odds, n_bookmakers) or None."""
    home_name = event.get("home_team", "")
    away_name = event.get("away_team", "")
    home_implied: list[float] = []
    away_implied: list[float] = []
    home_prices: list[float] = []
    away_prices: list[float] = []
    bookmakers = event.get("bookmakers") or []

    for book in bookmakers:
        for market in book.get("markets") or []:
            if market.get("key") != "h2h":
                continue
            home_odds = away_odds = None
            for outcome in market.get("outcomes") or []:
                price = outcome.get("price")
                if price is None:
                    continue
                try:
                    decimal = float(price)
                except (TypeError, ValueError):
                    continue
                name = outcome.get("name", "")
                if _teams_match(name, home_name):
                    home_odds = decimal
                elif _teams_match(name, away_name):
                    away_odds = decimal
            if home_odds and away_odds and home_odds > 1.0 and away_odds > 1.0:
                home_implied.append(_implied_from_decimal(home_odds))
                away_implied.append(_implied_from_decimal(away_odds))
                home_prices.append(home_odds)
                away_prices.append(away_odds)

    if not home_implied:
        return None

    avg_home_raw = sum(home_implied) / len(home_implied)
    avg_away_raw = sum(away_implied) / len(away_implied)
    home_p, away_p = _devig_two_way(avg_home_raw, avg_away_raw)
    avg_home_odds = sum(home_prices) / len(home_prices)
    avg_away_odds = sum(away_prices) / len(away_prices)
    return home_p, away_p, avg_home_odds, avg_away_odds, len(home_implied)


def _fetch_odds_api_events() -> list[dict[str, Any]]:
    if not ODDS_API_KEY:
        return []

    now = time.time()
    if now - _odds_cache["fetched_at"] <= _ODDS_CACHE_TTL and _odds_cache["events"]:
        return _odds_cache["events"]

    url = f"{_ODDS_API_BASE}/sports/{_ODDS_SPORT}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": _ODDS_REGIONS,
        "markets": _ODDS_MARKETS,
        "oddsFormat": "decimal",
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                url,
                params=params,
                headers={"User-Agent": SQUIGGLE_USER_AGENT},
            )
            response.raise_for_status()
            events = response.json()
            if isinstance(events, list):
                _odds_cache["events"] = events
                _odds_cache["fetched_at"] = now
                return events
    except Exception as exc:
        logger.warning("Odds API fetch failed: %s", exc)
    return []


def _find_odds_event(events: list[dict[str, Any]], match: Match) -> dict[str, Any] | None:
    for event in events:
        if _teams_match(event.get("home_team", ""), match.home_team) and _teams_match(
            event.get("away_team", ""), match.away_team
        ):
            return event
    return None


def _odds_api_snapshot(match: Match) -> MarketSnapshot | None:
    events = _fetch_odds_api_events()
    event = _find_odds_event(events, match)
    if not event:
        return None
    parsed = _parse_odds_event(event)
    if not parsed:
        return None
    home_p, away_p, home_odds, away_odds, n_books = parsed
    return MarketSnapshot(
        source="odds_api",
        home_implied_prob=_pct(home_p),
        away_implied_prob=_pct(away_p),
        home_decimal_odds=round(home_odds, 3),
        away_decimal_odds=round(away_odds, 3),
        n_bookmakers=n_books,
        note=f"Averaged across {n_books} bookmaker price(s), vig removed.",
    )


def _mock_snapshot(match: Match, model_home_pct: float) -> MarketSnapshot:
    """Deterministic demo market anchored slightly below model (for UI/dev)."""
    # Stable offset from match id so demos are reproducible.
    offset = ((match.id * 17) % 11 - 5) * 0.4  # roughly -2 .. +2 pts
    home_pct = max(5.0, min(95.0, model_home_pct - 2.5 + offset))
    away_pct = 100.0 - home_pct
    home_odds = round(100.0 / home_pct, 2) if home_pct > 0 else None
    away_odds = round(100.0 / away_pct, 2) if away_pct > 0 else None
    return MarketSnapshot(
        source="mock",
        home_implied_prob=round(home_pct, 2),
        away_implied_prob=round(away_pct, 2),
        home_decimal_odds=home_odds,
        away_decimal_odds=away_odds,
        note="Synthetic market for demo — configure ODDS_API_KEY or ingest Squiggle tips.",
    )


def fetch_market_snapshot(
    session: Session,
    match: Match,
    *,
    model_home_pct: float = 50.0,
) -> MarketSnapshot:
    """Resolve the best available market snapshot for ``match``."""
    snapshot = _odds_api_snapshot(match)
    if snapshot:
        return snapshot

    snapshot = _squiggle_consensus(session, match)
    if snapshot:
        # Derive fair decimal odds from consensus probs for Kelly sizing.
        if snapshot.home_implied_prob > 0:
            snapshot.home_decimal_odds = round(100.0 / snapshot.home_implied_prob, 3)
        if snapshot.away_implied_prob > 0:
            snapshot.away_decimal_odds = round(100.0 / snapshot.away_implied_prob, 3)
        return snapshot

    return _mock_snapshot(match, model_home_pct)


def _kelly_fraction(model_prob_0_1: float, decimal_odds: float) -> float:
    """Full Kelly stake fraction for a single outcome (returns 0 if no edge)."""
    if decimal_odds <= 1.0:
        return 0.0
    b = decimal_odds - 1.0
    p = model_prob_0_1
    q = 1.0 - p
    kelly = (b * p - q) / b
    return max(0.0, kelly)


def _recommendation(
    bet_side: str,
    edge_pct: float,
    model_pct: float,
    market_pct: float,
    team_name: str,
) -> str:
    if bet_side == "none":
        return "Pass — model and market agree within 2% (no actionable edge)."

    magnitude = abs(edge_pct)
    direction = "Value" if magnitude >= 5.0 else "Lean"
    sign = "+" if edge_pct >= 0 else ""
    return (
        f"{direction} {team_name} — model {model_pct:.1f}% vs market "
        f"{market_pct:.1f}% ({sign}{edge_pct:.1f}% edge)."
    )


def compare_model_to_market(
    model_home_win_prob: float,
    market: MarketSnapshot,
    *,
    match_id: int = 0,
    home_team: str = "",
    away_team: str = "",
) -> MarketEdgeResult:
    """Compare model probabilities (0–100 scale) to a market snapshot."""
    model_home = float(model_home_win_prob)
    model_away = 100.0 - model_home

    home_edge = model_home - market.home_implied_prob
    away_edge = model_away - market.away_implied_prob

    if home_edge >= away_edge and home_edge > 2.0:
        bet_side: Literal["home", "away", "none"] = "home"
        edge_pct = home_edge
        model_pct = model_home
        market_pct = market.home_implied_prob
        decimal_odds = market.home_decimal_odds
        team = home_team or "home"
        model_prob_0_1 = model_home / 100.0
    elif away_edge > home_edge and away_edge > 2.0:
        bet_side = "away"
        edge_pct = away_edge
        model_pct = model_away
        market_pct = market.away_implied_prob
        decimal_odds = market.away_decimal_odds
        team = away_team or "away"
        model_prob_0_1 = model_away / 100.0
    else:
        bet_side = "none"
        if abs(home_edge) >= abs(away_edge):
            edge_pct = home_edge
            model_pct = model_home
            market_pct = market.home_implied_prob
        else:
            edge_pct = away_edge
            model_pct = model_away
            market_pct = market.away_implied_prob
        decimal_odds = None
        team = ""
        model_prob_0_1 = 0.0

    kelly = 0.0
    if bet_side != "none" and decimal_odds:
        kelly = _kelly_fraction(model_prob_0_1, decimal_odds) / 4.0  # quarter Kelly

    rec = _recommendation(bet_side, edge_pct, model_pct, market_pct, team)

    return MarketEdgeResult(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        model_home_win_prob=model_home,
        model_away_win_prob=model_away,
        market=market,
        edge_pct=edge_pct,
        bet_side=bet_side,
        kelly_fraction=kelly,
        recommendation=rec,
    )


def get_market_edge(
    session: Session,
    match: Match,
    *,
    stored: StoredPrediction | None = None,
) -> dict[str, Any]:
    """Full model-vs-market edge payload for API consumption."""
    if stored is None:
        stored = session.scalars(
            select(StoredPrediction).where(StoredPrediction.match_id == match.id)
        ).first()

    model_home = stored.home_win_prob if stored else 50.0
    market = fetch_market_snapshot(session, match, model_home_pct=model_home)
    result = compare_model_to_market(
        model_home,
        market,
        match_id=match.id,
        home_team=match.home_team,
        away_team=match.away_team,
    )
    return result.to_dict()


def get_round_market_edges(
    session: Session,
    year: int,
    round_num: int,
) -> dict[str, Any]:
    """Market edges for every match in a round."""
    matches = session.scalars(
        select(Match)
        .where(Match.year == year, Match.round == round_num)
        .order_by(Match.date, Match.id)
    ).all()

    stored_by_match = {
        row.match_id: row
        for row in session.scalars(
            select(StoredPrediction).where(
                StoredPrediction.match_id.in_([m.id for m in matches])
            )
        ).all()
    }

    edges = [
        get_market_edge(session, match, stored=stored_by_match.get(match.id))
        for match in matches
    ]
    return {"year": year, "round": round_num, "edges": edges}
