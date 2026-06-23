"""Offline batch builder: precompute and store match predictions.

For each requested season this fits the Elo/Ridge/logistic models ONCE (via the
per-year memo in :mod:`src.predict.service`) and upserts a
:class:`~src.db.models.StoredPrediction` row for *every* match in that season
(all rounds, completed or not). The API then serves these rows with a plain
indexed read instead of refitting the models on every request.

The script is idempotent — re-running upserts on ``match_id`` — so it is safe to
run repeatedly (e.g. after ingesting new results).

Detail (margin histogram + per-player Monte Carlo projections) is intentionally
NOT precomputed here: running 25k-sim player simulations for every historical
match would take far too long, and the dashboard's slow path was the round view,
not the detail view. ``StoredPrediction.detail_json`` is left NULL; the
``/predict/{id}`` endpoint computes detail on first access and back-fills it.

Usage::

    python scripts/build_predictions.py 2024 2025 2026
    python scripts/build_predictions.py            # defaults to 2024 2025 2026
"""

from __future__ import annotations

import sys
import time

from src.db.session import get_session, init_db
from src.predict import clear_cache, predict_season, upsert_stored_prediction

DEFAULT_SEASONS = [2024, 2025, 2026]


def build(seasons: list[int]) -> int:
    init_db()
    # Start from a clean memo so each run reflects the current DB contents.
    clear_cache()

    total = 0
    for season in seasons:
        start = time.perf_counter()
        with get_session() as session:
            items = predict_season(session, season)
            for item in items:
                upsert_stored_prediction(session, item)
            # get_session commits on exit.
        elapsed = time.perf_counter() - start
        total += len(items)
        print(
            f"[{season}] wrote {len(items)} predictions in {elapsed:.1f}s "
            f"(cumulative {total})"
        )

    print(f"Done. {total} predictions stored across {len(seasons)} season(s).")
    return total


def main() -> None:
    seasons = [int(a) for a in sys.argv[1:]] or DEFAULT_SEASONS
    build(seasons)


if __name__ == "__main__":
    main()
