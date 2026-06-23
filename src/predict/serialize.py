"""Lightweight (de)serialization helpers for stored predictions.

This module deliberately imports **only** SQLAlchemy (already required to serve)
— no numpy/pandas/scikit-learn/numba. The serving API uses these helpers on its
fast paths (reading precomputed :class:`StoredPrediction` rows / ``detail_json``)
so that a request never drags in the heavy modelling stack. The expensive model
code lives in :mod:`src.predict.service` and is imported lazily only when a
prediction actually has to be *computed* (the offline build path / fallbacks).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

# Provenance label for the headline win-prob model (logistic base + sigmoid
# Platt calibration). Kept here so the serving path can reference it without
# importing the modelling stack.
WIN_PROB_SOURCE = "logistic+sigmoid"


def upsert_stored_prediction(
    session: Session, item: dict[str, Any], *, model_version: str = "hybrid_v1"
):
    """Insert or update the :class:`StoredPrediction` row for ``item``.

    ``item`` is a compact row produced by the prediction service. Idempotent:
    keyed on ``match_id``. Does NOT commit (the caller controls the transaction).
    """
    from src.db.models import StoredPrediction

    row = session.scalars(
        select(StoredPrediction).where(
            StoredPrediction.match_id == item["match_id"]
        )
    ).first()
    if row is None:
        row = StoredPrediction(match_id=item["match_id"])
        session.add(row)

    row.model_version = model_version
    row.win_prob_source = item.get("win_prob_source", WIN_PROB_SOURCE)
    row.margin_source = item.get("margin_source", "ridge")
    row.home_win_prob = float(item["home_win_prob"])
    row.away_win_prob = float(item["away_win_prob"])
    row.predicted_winner = item["predicted_winner"]
    row.confidence = float(item["confidence"])
    row.predicted_margin = float(item["predicted_margin"])
    row.predicted_home_score = float(item["predicted_home_score"])
    row.predicted_away_score = float(item["predicted_away_score"])
    return row


def stored_to_item(row, match) -> dict[str, Any]:
    """Render a :class:`StoredPrediction` row back into the compact API shape.

    ``match`` provides the live score/complete fields so completed-game results
    stay current even if predictions were stored earlier.
    """
    return {
        "match_id": row.match_id,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "date": match.date.isoformat() if match.date else None,
        "venue": match.venue,
        "complete": bool(match.complete),
        "home_score": match.home_score,
        "away_score": match.away_score,
        "predicted_winner": row.predicted_winner,
        "home_win_prob": row.home_win_prob,
        "away_win_prob": row.away_win_prob,
        "predicted_margin": row.predicted_margin,
        "predicted_home_score": row.predicted_home_score,
        "predicted_away_score": row.predicted_away_score,
        "confidence": row.confidence,
    }
