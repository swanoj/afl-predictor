"""Build global conformal calibration quantiles from walk-forward backtests.

Collects absolute residuals ``|outcome - predicted_prob|`` from the
leakage-free calibrated logistic model across completed seasons, then writes
``deploy/conformal.json`` for :mod:`src.predict.conformal`.

Numpy is used here only in this offline script (not at serving time).

Usage::

    python scripts/build_conformal.py
    python scripts/build_conformal.py --seasons 2019 2020 2021 2022 2023 2024 2025
    python scripts/build_conformal.py --out deploy/conformal.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import ROOT_DIR
from src.db.session import get_session, init_db
from src.eval.backtest import calibrated_match_predictions

DEFAULT_OUT = ROOT_DIR / "deploy" / "conformal.json"
DEFAULT_SEASONS = list(range(2019, 2026))
DEFAULT_COVERAGE = 0.9
MODEL_VERSION = "calibrated:logistic:sigmoid"
CALIB_KIND = "logistic"
CALIB_METHOD = "sigmoid"


def _conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    """Finite-sample conformal quantile for symmetric intervals."""
    if residuals.size == 0:
        return 0.45
    alpha = 1.0 - coverage
    # ceil((n+1)(1-alpha)) / n quantile index for split conformal.
    n = residuals.size
    k = int(np.ceil((n + 1) * (1.0 - alpha / 2.0)))
    k = min(max(k, 1), n)
    return float(np.sort(residuals)[k - 1])


def build(
    seasons: list[int],
    *,
    coverage: float = DEFAULT_COVERAGE,
    out_path: Path = DEFAULT_OUT,
) -> dict:
    init_db()
    all_residuals: list[float] = []
    by_year: dict[str, dict] = {}

    with get_session() as session:
        for season in seasons:
            try:
                preds = calibrated_match_predictions(
                    session, season, CALIB_KIND, CALIB_METHOD
                )
            except ValueError as exc:
                print(f"  skip {season}: {exc}")
                continue

            residuals = np.array(
                [abs(p.outcome - p.home_win_prob) for p in preds], dtype=float
            )
            if residuals.size == 0:
                print(f"  skip {season}: no predictions")
                continue

            all_residuals.extend(residuals.tolist())
            by_year[str(season)] = {
                "residual_quantile": _conformal_quantile(residuals, coverage),
                "n": int(residuals.size),
            }
            print(f"  {season}: n={residuals.size}, q={by_year[str(season)]['residual_quantile']:.4f}")

    residual_arr = np.array(all_residuals, dtype=float)
    global_q = _conformal_quantile(residual_arr, coverage)

    payload = {
        "version": 1,
        "model_version": MODEL_VERSION,
        "coverage": coverage,
        "quantile_level": round(1.0 - (1.0 - coverage) / 2.0, 4),
        "residual_quantile": global_q,
        "n_calibration": int(residual_arr.size),
        "seasons": sorted(int(y) for y in by_year),
        "by_year": by_year,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(
        f"Wrote {out_path} — global q={global_q:.4f} "
        f"from n={payload['n_calibration']} walk-forward games"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Seasons to include in calibration (default: 2019-2025).",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=DEFAULT_COVERAGE,
        help="Target interval coverage (default: 0.9).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path (default: deploy/conformal.json).",
    )
    args = parser.parse_args()
    build(args.seasons, coverage=args.coverage, out_path=args.out)


if __name__ == "__main__":
    main()
