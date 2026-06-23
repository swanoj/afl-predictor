"""FastAPI REST API for AFL predictions."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func, select

from src.config import DEFAULT_SIMS, ROOT_DIR
from src.db.models import Match, SimulationRun, StoredPrediction
from src.db.session import SessionLocal, init_db

# NOTE: only the *lightweight* serialization helpers are imported eagerly. The
# heavy modelling stack (numpy/pandas/scikit-learn/numba) lives in
# ``src.predict.service`` and is imported lazily, inside the fallback branches
# that actually have to *compute* a prediction. In production everything is
# precomputed (StoredPrediction rows + detail_json), so the request hot paths
# never import the modelling stack at all.
from src.intelligence.market import get_market_edge, get_round_market_edges
from src.intelligence.service import get_match_intelligence, get_round_intelligence
from src.intelligence.similar_games import find_similar_games
from src.predict.conformal import get_conformal_interval
from src.predict.enriched import enrich_prediction_item, run_whatif
from src.predict.serialize import stored_to_item, upsert_stored_prediction


class SimulateRequest(BaseModel):
    pressure_index: float | None = None
    territory_tilt: float | None = None
    home_momentum: float | None = None
    away_momentum: float | None = None
    n_sims: int = DEFAULT_SIMS


class EnvironmentOverride(BaseModel):
    pressure_index: float = 0.5
    territory_tilt: float = 0.0
    home_momentum: float = 0.0
    away_momentum: float = 0.0


class WhatIfRequest(BaseModel):
    home_out: list[str] = []
    away_out: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AFL God-Tier Predictor", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/matches")
def list_matches(year: int = 2026, round: int | None = None) -> list[dict[str, Any]]:
    session = SessionLocal()
    try:
        query = select(Match).where(Match.year == year)
        if round is not None:
            query = query.where(Match.round == round)
        matches = session.scalars(query.order_by(Match.round)).all()
        return [
            {
                "id": m.id,
                "squiggle_id": m.squiggle_id,
                "year": m.year,
                "round": m.round,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "home_score": m.home_score,
                "away_score": m.away_score,
                "complete": m.complete,
                "venue": m.venue,
            }
            for m in matches
        ]
    finally:
        session.close()


@app.get("/predict-round")
def predict_round(year: int = 2026, round: int = 1) -> dict[str, Any]:
    """Compact predictions for every game in a round, in one call.

    Fast path: read precomputed :class:`StoredPrediction` rows (written offline
    by ``scripts/build_predictions.py``) — a plain indexed read, instant even on
    a cold server. Fallback: if any of the round's matches lack a stored row,
    fit the models once, compute the whole round, store the rows, and return.
    """
    session = SessionLocal()
    try:
        matches = session.scalars(
            select(Match)
            .where(Match.year == year, Match.round == round)
            .order_by(Match.date, Match.id)
        ).all()

        stored = {
            row.match_id: row
            for row in session.scalars(
                select(StoredPrediction).where(
                    StoredPrediction.match_id.in_([m.id for m in matches])
                )
            ).all()
        }

        if matches and all(m.id in stored for m in matches):
            predictions = [
                enrich_prediction_item(session, m, stored_to_item(stored[m.id], m))
                for m in matches
            ]
        else:
            # Fallback: compute (fits models) + persist for next time. Heavy
            # modelling stack imported lazily so the precomputed hot path stays
            # free of numpy/pandas/sklearn/numba.
            from src.predict.service import predict_round as run_round_prediction

            predictions = run_round_prediction(session, year, round)
            for item in predictions:
                upsert_stored_prediction(session, item)
            session.commit()

        return {
            "year": year,
            "round": round,
            "win_prob_source": "logistic+sigmoid",
            "margin_source": "ridge",
            "predictions": predictions,
        }
    finally:
        session.close()


@app.get("/predictions/status")
def predictions_status() -> dict[str, Any]:
    """Coverage of stored predictions per season (stored vs total matches)."""
    session = SessionLocal()
    try:
        match_counts = dict(
            session.execute(
                select(Match.year, func.count(Match.id)).group_by(Match.year)
            ).all()
        )
        stored_counts = dict(
            session.execute(
                select(Match.year, func.count(StoredPrediction.id))
                .join(StoredPrediction, StoredPrediction.match_id == Match.id)
                .group_by(Match.year)
            ).all()
        )
        seasons = sorted(set(match_counts) | set(stored_counts))
        return {
            "total_stored": int(sum(stored_counts.values())),
            "seasons": [
                {
                    "year": int(year),
                    "stored": int(stored_counts.get(year, 0)),
                    "matches": int(match_counts.get(year, 0)),
                }
                for year in seasons
            ],
        }
    finally:
        session.close()


@app.get("/intelligence/round")
def intelligence_round(year: int = 2026, round: int = 1) -> dict[str, Any]:
    """Live AFL news, injuries, and per-match intel for a round."""
    session = SessionLocal()
    try:
        return get_round_intelligence(session, year, round)
    finally:
        session.close()


@app.get("/intelligence/market/{match_id}")
def intelligence_market(match_id: int) -> dict[str, Any]:
    """Model vs market edge, Kelly fraction, and betting recommendation."""
    session = SessionLocal()
    try:
        match = session.get(Match, match_id)
        if not match:
            raise HTTPException(404, "Match not found")
        return get_market_edge(session, match)
    finally:
        session.close()


@app.get("/intelligence/market")
def intelligence_market_round(year: int = 2026, round: int = 1) -> dict[str, Any]:
    """Model vs market edges for every match in a round."""
    session = SessionLocal()
    try:
        return get_round_market_edges(session, year, round)
    finally:
        session.close()


@app.get("/intelligence/match/{match_id}")
def intelligence_match(match_id: int) -> dict[str, Any]:
    """News, injuries, sentiment, and AI briefing for one match."""
    session = SessionLocal()
    try:
        match = session.get(Match, match_id)
        if not match:
            raise HTTPException(404, "Match not found")
        return get_match_intelligence(session, match)
    finally:
        session.close()


@app.get("/intelligence/similar-games/{match_id}")
def intelligence_similar_games(match_id: int) -> dict[str, Any]:
    """Top historical matches with a similar macro profile (RAG context)."""
    session = SessionLocal()
    try:
        match = session.get(Match, match_id)
        if not match:
            raise HTTPException(404, "Match not found")
        return find_similar_games(session, match)
    finally:
        session.close()


@app.get("/predict/{match_id}/interval")
def predict_interval(match_id: int) -> dict[str, Any]:
    """90% conformal interval for home win probability."""
    session = SessionLocal()
    try:
        match = session.get(Match, match_id)
        if not match:
            raise HTTPException(404, "Match not found")

        stored = session.scalars(
            select(StoredPrediction).where(StoredPrediction.match_id == match_id)
        ).first()
        if stored is None:
            raise HTTPException(404, "No stored prediction for this match")

        interval = get_conformal_interval(stored.home_win_prob, match.year)
        return {
            "match_id": match_id,
            "home_win_prob": stored.home_win_prob,
            "year": match.year,
            **interval,
        }
    finally:
        session.close()


@app.post("/predict/{match_id}/whatif")
def predict_whatif(match_id: int, body: WhatIfRequest) -> dict[str, Any]:
    """Counterfactual win probability with selected players OUT."""
    session = SessionLocal()
    try:
        match = session.get(Match, match_id)
        if not match:
            raise HTTPException(404, "Match not found")
        try:
            return run_whatif(
                session,
                match,
                home_out=body.home_out,
                away_out=body.away_out,
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
    finally:
        session.close()


@app.get("/predict/{match_id}")
def predict_match(match_id: int, n_sims: int = DEFAULT_SIMS) -> dict[str, Any]:
    session = SessionLocal()
    try:
        match = session.get(Match, match_id)
        if not match:
            raise HTTPException(404, "Match not found")

        stored = session.scalars(
            select(StoredPrediction).where(StoredPrediction.match_id == match_id)
        ).first()

        # Fast path: full detail was cached on a prior request.
        if stored is not None and stored.detail_json:
            return json.loads(stored.detail_json)

        # Compute full detail (win prob = calibrated logistic; margin/scores =
        # Ridge; player projections = Monte Carlo simulator), then cache it.
        # Heavy modelling stack imported lazily — in production the line above
        # always returns from detail_json, so this never runs / never imports.
        from src.predict.service import predict_match as run_prediction

        prediction = run_prediction(session, match, n_sims=n_sims)

        if stored is not None:
            stored.detail_json = json.dumps(prediction)
        else:
            # No summary row yet (build script not run for this season): create
            # one so the round view is instant too, and cache the detail.
            home_wp = prediction["home_win_prob"]
            away_wp = prediction["away_win_prob"]
            upsert_stored_prediction(
                session,
                {
                    "match_id": match.id,
                    "home_win_prob": home_wp,
                    "away_win_prob": away_wp,
                    "predicted_winner": (
                        match.home_team if home_wp >= away_wp else match.away_team
                    ),
                    "confidence": max(home_wp, away_wp),
                    "predicted_margin": prediction["median_margin"],
                    "predicted_home_score": prediction["median_home_score"],
                    "predicted_away_score": prediction["median_away_score"],
                    "win_prob_source": prediction["win_prob_source"],
                    "margin_source": prediction["margin_source"],
                },
            )
            new_row = session.scalars(
                select(StoredPrediction).where(
                    StoredPrediction.match_id == match.id
                )
            ).first()
            if new_row is not None:
                new_row.detail_json = json.dumps(prediction)

        session.add(
            SimulationRun(
                match_id=match.id,
                n_sims=n_sims,
                home_win_prob=prediction["home_win_prob"],
                away_win_prob=prediction["away_win_prob"],
                median_home_score=prediction["median_home_score"],
                median_away_score=prediction["median_away_score"],
                median_margin=prediction["median_margin"],
                model_version=prediction["win_prob_source"],
            )
        )
        session.commit()

        return prediction
    finally:
        session.close()


@app.post("/simulate/{match_id}")
def simulate_with_overrides(match_id: int, body: SimulateRequest) -> dict[str, Any]:
    session = SessionLocal()
    try:
        match = session.get(Match, match_id)
        if not match:
            raise HTTPException(404, "Match not found")

        # Legacy endpoint. Environment overrides only steer the player
        # projections; the headline win prob (calibrated logistic) and margin
        # (Ridge) are feature-driven (rolling team form + Elo), not the
        # EnvironmentState, so they are unchanged by the sliders. The sliders
        # have been removed from the product as misleading — this endpoint is
        # kept only for backward compatibility.
        env_overrides = {
            "pressure_index": body.pressure_index,
            "territory_tilt": body.territory_tilt,
            "home_momentum": body.home_momentum,
            "away_momentum": body.away_momentum,
        }
        # Legacy compute-on-demand path: heavy stack imported lazily.
        from src.predict.service import predict_match as run_prediction

        prediction = run_prediction(
            session, match, n_sims=body.n_sims, env_overrides=env_overrides
        )

        return {
            "match_id": prediction["match_id"],
            "environment": prediction["environment"],
            "home_win_prob": prediction["home_win_prob"],
            "away_win_prob": prediction["away_win_prob"],
            "median_home_score": prediction["median_home_score"],
            "median_away_score": prediction["median_away_score"],
            "median_margin": prediction["median_margin"],
            "std_margin": prediction["std_margin"],
            "p95_margin": prediction["p95_margin"],
            "margin_histogram": prediction["margin_histogram"],
            "player_projections": prediction["player_projections"],
            "win_prob_source": prediction["win_prob_source"],
        }
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Static frontend (single-origin SPA hosting)
#
# Serve the built Vite app (``frontend/dist``) from the SAME FastAPI service so
# the whole dashboard lives behind one Render URL. The API routes above are
# registered first, so they always take precedence; this catch-all only handles
# everything else (the SPA shell + its hashed assets). Registered last on
# purpose.
# ---------------------------------------------------------------------------
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

# Known API path prefixes — a request that falls through to the catch-all with
# one of these prefixes is a genuine 404, not an SPA route.
_API_PREFIXES = {
    "health",
    "matches",
    "predict",
    "predict-round",
    "predictions",
    "intelligence",
    "simulate",
    "docs",
    "redoc",
    "openapi.json",
}

if FRONTEND_DIST.is_dir():
    _assets_dir = FRONTEND_DIST / "assets"
    if _assets_dir.is_dir():
        app.mount(
            "/assets", StaticFiles(directory=str(_assets_dir)), name="assets"
        )

    @app.get("/", include_in_schema=False)
    def _spa_root() -> FileResponse:
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str) -> FileResponse:
        first = full_path.split("/", 1)[0]
        if first in _API_PREFIXES:
            raise HTTPException(404, "Not found")

        # Serve a real file if it exists (favicon, etc.), guarding against path
        # traversal; otherwise return index.html for client-side routing.
        candidate = (FRONTEND_DIST / full_path).resolve()
        if FRONTEND_DIST.resolve() in candidate.parents and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
