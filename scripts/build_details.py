"""Offline batch builder: precompute the FULL per-match detail.

For every match this computes the rich ``/predict/{id}`` payload — the margin
histogram plus per-player Monte Carlo disposal/goal projections — and caches it
as JSON in :attr:`StoredPrediction.detail_json`. Once this has run, the
``/predict/{id}`` endpoint serves detail with a plain ``json.loads`` of a
column: **no Elo/Ridge/logistic refit, no numba JIT, no scikit-learn at request
time.**

Models are fit ONCE per season (via the per-year memo in
:mod:`src.predict.service`); the script then loops every match in that season.
The Monte Carlo is reduced to ``--n-sims`` (default 2000) because the detail is
illustrative — the headline numbers (win prob / margin / scores) are the
calibrated logistic + Ridge point estimates stored separately and are unchanged
by this script.

This must run against the FULL database (player_game_logs etc. are required to
build rosters); the lean serving DB is exported afterwards by
``scripts/export_serving_db.py``.

Usage::

    python scripts/build_details.py                  # 2024 2025 2026, n_sims=2000
    python scripts/build_details.py 2026             # one season
    python scripts/build_details.py --n-sims 1000 2026
"""

from __future__ import annotations

import argparse
import json
import time

from sqlalchemy import select

from src.db.models import Match, StoredPrediction
from src.db.session import get_session, init_db
from src.predict import clear_cache, predict_match, upsert_stored_prediction

DEFAULT_SEASONS = [2024, 2025, 2026]
DEFAULT_N_SIMS = 2000


def build(seasons: list[int], n_sims: int) -> tuple[int, int]:
    init_db()
    # Fresh memo so each run reflects current DB contents.
    clear_cache()

    total_rows = 0
    total_failures = 0
    grand_start = time.perf_counter()

    for season in seasons:
        start = time.perf_counter()
        season_rows = 0
        season_fail = 0
        with get_session() as session:
            matches = session.scalars(
                select(Match)
                .where(Match.year == season)
                .order_by(Match.round, Match.date, Match.id)
            ).all()

            # Map existing stored summary rows so we can attach detail_json.
            stored = {
                row.match_id: row
                for row in session.scalars(
                    select(StoredPrediction).where(
                        StoredPrediction.match_id.in_([m.id for m in matches])
                    )
                ).all()
            }

            for match in matches:
                try:
                    detail = predict_match(session, match, n_sims=n_sims)
                except Exception as exc:  # pragma: no cover - defensive
                    season_fail += 1
                    print(f"  ! match {match.id} ({season} R{match.round}) "
                          f"failed: {exc}")
                    continue

                row = stored.get(match.id)
                if row is None:
                    # No summary row yet — create one from the detail headline so
                    # the round view is instant too, then attach the detail.
                    home_wp = detail["home_win_prob"]
                    away_wp = detail["away_win_prob"]
                    upsert_stored_prediction(
                        session,
                        {
                            "match_id": match.id,
                            "home_win_prob": home_wp,
                            "away_win_prob": away_wp,
                            "predicted_winner": (
                                match.home_team
                                if home_wp >= away_wp
                                else match.away_team
                            ),
                            "confidence": max(home_wp, away_wp),
                            "predicted_margin": detail["median_margin"],
                            "predicted_home_score": detail["median_home_score"],
                            "predicted_away_score": detail["median_away_score"],
                            "win_prob_source": detail["win_prob_source"],
                            "margin_source": detail["margin_source"],
                        },
                    )
                    row = session.scalars(
                        select(StoredPrediction).where(
                            StoredPrediction.match_id == match.id
                        )
                    ).first()

                if row is not None:
                    row.detail_json = json.dumps(detail)
                    season_rows += 1
            # get_session commits on exit.

        elapsed = time.perf_counter() - start
        total_rows += season_rows
        total_failures += season_fail
        print(
            f"[{season}] detail for {season_rows} matches in {elapsed:.1f}s "
            f"({season_fail} failures, cumulative {total_rows})"
        )

    grand = time.perf_counter() - grand_start
    print(
        f"Done. {total_rows} detail rows across {len(seasons)} season(s) "
        f"in {grand:.1f}s ({total_failures} failures), n_sims={n_sims}."
    )
    return total_rows, total_failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "seasons",
        nargs="*",
        type=int,
        default=DEFAULT_SEASONS,
        help="Seasons to build detail for (default: 2024 2025 2026).",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=DEFAULT_N_SIMS,
        help="Monte Carlo simulations per match for detail (default: 2000).",
    )
    args = parser.parse_args()
    seasons = args.seasons or DEFAULT_SEASONS
    build(seasons, args.n_sims)


if __name__ == "__main__":
    main()
