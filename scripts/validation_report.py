"""Consolidated validation report for the AFL predictor.

Single source of truth for "is this model real?". Runs, for 2024 and 2025:

  * walk-forward backtests for every model (Brier / MAE / accuracy / ECE)
  * the Squiggle consensus benchmark (do we beat the market?)
  * calibration / reliability + Expected Calibration Error (before vs after)

Models evaluated:
  * elo_v1                       — Elo logistic win prob (baseline)
  * ridge_v1                     — Ridge margin -> norm.cdf(margin/sigma) (baseline)
  * logistic_v1                  — direct L2 logistic on the macro features
  * gbm_v1                       — HistGradientBoosting on the macro features
  * calibrated:{base}:{method}   — best base recalibrated (isotonic / sigmoid)

Saves a reliability-curve PNG and prints one consolidated, honest report.

Usage:
    python scripts/validation_report.py [--no-cache] [--seasons 2024 2025]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.config import DATA_DIR
from src.db.session import get_session, init_db
from src.eval.backtest import match_predictions
from src.eval.calibration import (
    CalibrationResult,
    plot_reliability,
    reliability_table,
)
from src.eval.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    mae,
    maximum_calibration_error,
)
from src.eval.squiggle_benchmark import (
    available_tip_seasons,
    benchmark_model_vs_consensus,
    format_benchmark,
)

# Order matters: baselines first, then the new direct/calibrated models, then
# the stacked integrated model (upgraded-Elo + enriched features + logistic +
# calibration) that combines all four parallel upgrades.
MODELS = [
    "elo_v1",
    "ridge_v1",
    "logistic_v1",
    "gbm_v1",
    "calibrated:logistic:isotonic",
    "calibrated:logistic:sigmoid",
    "calibrated:gbm:isotonic",
    "calibrated:gbm:sigmoid",
    "integrated_v1",
]

# Short labels for tables.
LABELS = {
    "elo_v1": "elo_v1",
    "ridge_v1": "ridge_v1",
    "logistic_v1": "logistic",
    "gbm_v1": "gbm",
    "calibrated:logistic:isotonic": "log+iso",
    "calibrated:logistic:sigmoid": "log+sig",
    "calibrated:gbm:isotonic": "gbm+iso",
    "calibrated:gbm:sigmoid": "gbm+sig",
    "integrated_v1": "integ_v1",
}

# Integrated-model feature ablations: each removes ONE added signal so we can
# attribute the integrated model's score. Labels describe the change.
ABLATIONS = [
    ("integrated_v1", "full (upgraded-elo + avail + xscore)"),
    ("integrated:flat_elo", "− upgraded-elo (use flat HGA Elo)"),
    ("integrated:drop_avail", "− avail_diff"),
    ("integrated:drop_xscore", "− xscore_diff"),
]


def _hr(char: str = "=") -> str:
    return char * 78


def eval_model(session, season: int, model_version: str) -> dict:
    """Run the walk-forward harness once and derive every scalar metric."""
    preds = match_predictions(session, season, model_version)
    probs = np.array([p.home_win_prob for p in preds])
    outcomes = np.array([p.outcome for p in preds])
    margin_preds = np.array([p.margin_pred for p in preds])
    margin_actuals = np.array([p.margin_actual for p in preds])
    return {
        "model": model_version,
        "season": season,
        "n": len(preds),
        "brier": brier_score(probs, outcomes),
        "mae": mae(margin_preds, margin_actuals),
        "logloss": log_loss(probs, outcomes),
        "accuracy": float(np.mean((probs >= 0.5) == outcomes)),
        "ece": expected_calibration_error(probs, outcomes),
        "mce": maximum_calibration_error(probs, outcomes),
        "_probs": probs,
        "_outcomes": outcomes,
    }


def _fmt_mae(v: float) -> str:
    return "   n/a" if np.isnan(v) else f"{v:>6.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="AFL validation report")
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2024, 2025],
        help="Seasons to evaluate (default: 2024 2025)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-fetch of Squiggle tips instead of using the DB cache",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Use a single Squiggle source (e.g. 'Squiggle') instead of consensus",
    )
    args = parser.parse_args()
    use_cache = not args.no_cache

    init_db()

    report: dict = {
        "seasons": args.seasons,
        "models": MODELS,
        "backtests": {},
        "benchmark": {},
        "calibration": {},
        "ablation": {},
    }

    print(_hr())
    print("AFL PREDICTOR — VALIDATION REPORT")
    print(_hr())

    # ------------------------------------------------------------------
    # 1. Backtests — Brier / MAE / logloss / accuracy / ECE per model.
    # ------------------------------------------------------------------
    print("\n[1] WALK-FORWARD BACKTESTS (honest, no leakage)\n")
    print(
        f"  {'model':>10} {'season':>6} {'Brier':>8} {'MAE':>6} {'logloss':>8}"
        f" {'acc':>7} {'ECE':>7} {'MCE':>7} {'n':>5}"
    )
    print("  " + "-" * 76)
    evals: dict[int, dict[str, dict]] = {}
    calib_for_plot: list[CalibrationResult] = []
    with get_session() as session:
        for season in args.seasons:
            evals[season] = {}
            for mv in MODELS:
                ev = eval_model(session, season, mv)
                evals[season][mv] = ev
                report["backtests"].setdefault(str(season), {})[mv] = {
                    k: v for k, v in ev.items() if not k.startswith("_")
                }
                print(
                    f"  {LABELS[mv]:>10} {season:>6} {ev['brier']:>8.4f} {_fmt_mae(ev['mae'])}"
                    f" {ev['logloss']:>8.4f} {ev['accuracy']:>7.1%}"
                    f" {ev['ece']:>7.4f} {ev['mce']:>7.4f} {ev['n']:>5}"
                )
            print()

    # ------------------------------------------------------------------
    # 2. Squiggle consensus benchmark — do we beat the market?
    # ------------------------------------------------------------------
    print(_hr())
    print("[2] SQUIGGLE CONSENSUS BENCHMARK (do we beat the market?)\n")
    with get_session() as session:
        avail = available_tip_seasons(
            session, args.seasons, use_cache=use_cache, persist=True
        )
    print("  Tip availability (games with consensus):")
    for season, n in avail.items():
        print(f"    {season}: {n} games")
    print()

    print(
        f"  {'model':>10} {'season':>6} {'ourBrier':>9} {'consBrier':>10}"
        f" {'delta':>8} {'verdict':>9} {'rounds':>9}"
    )
    print("  " + "-" * 70)
    with get_session() as session:
        for season in args.seasons:
            for mv in MODELS:
                res = benchmark_model_vs_consensus(
                    session, season, mv,
                    source=args.source, use_cache=use_cache, persist=True,
                )
                report["benchmark"].setdefault(str(season), {})[mv] = res.to_dict()
                if not res.available:
                    print(f"  {LABELS[mv]:>10} {season:>6}   (no tips)")
                    continue
                delta = res.consensus_brier - res.our_brier
                verdict = "BEATS" if res.beats_consensus else "loses"
                print(
                    f"  {LABELS[mv]:>10} {season:>6} {res.our_brier:>9.4f}"
                    f" {res.consensus_brier:>10.4f} {delta:>+8.4f} {verdict:>9}"
                    f" {res.rounds_beaten:>3}/{res.rounds_total:<3}"
                )
            print()

    # ------------------------------------------------------------------
    # 3. Calibration / reliability — ECE before vs after recalibration.
    # ------------------------------------------------------------------
    print(_hr())
    print("[3] CALIBRATION / RELIABILITY (are our probabilities trustworthy?)\n")
    print("  ECE by model (lower is better). 'after' rows are recalibrated bases.\n")
    print(f"  {'model':>10} {'season':>6} {'ECE':>8} {'MCE':>8} {'Brier':>8}")
    print("  " + "-" * 46)
    for season in args.seasons:
        for mv in MODELS:
            ev = evals[season][mv]
            report["calibration"].setdefault(str(season), {})[mv] = {
                "ece": ev["ece"], "mce": ev["mce"], "brier": ev["brier"],
            }
            print(
                f"  {LABELS[mv]:>10} {season:>6} {ev['ece']:>8.4f}"
                f" {ev['mce']:>8.4f} {ev['brier']:>8.4f}"
            )
            # Collect a few reliability curves for the plot.
            if mv in ("ridge_v1", "logistic_v1", "calibrated:logistic:isotonic"):
                calib_for_plot.append(
                    CalibrationResult(
                        model_version=LABELS[mv],
                        season=season,
                        n_games=ev["n"],
                        ece=ev["ece"],
                        mce=ev["mce"],
                        brier=ev["brier"],
                        bins=reliability_table(ev["_probs"], ev["_outcomes"]),
                    )
                )
        print()

    png_path = plot_reliability(calib_for_plot, DATA_DIR / "calibration_reliability.png")
    print(f"  Reliability curve saved to: {png_path}")

    # ------------------------------------------------------------------
    # 3b. Ablation — what does each stacked upgrade actually contribute?
    # ------------------------------------------------------------------
    print("\n" + _hr())
    print("[3b] INTEGRATED ABLATION (does each added signal help?)\n")
    print("  Brier on the full season backtest. 'contrib' = Brier(this) − Brier(full):")
    print("  positive contrib means REMOVING that signal HURTS (i.e. it helps).\n")
    print(f"  {'season':>6} {'variant':>38} {'Brier':>8} {'ECE':>8} {'contrib':>9}")
    print("  " + "-" * 74)
    with get_session() as session:
        for season in args.seasons:
            full_brier = evals[season]["integrated_v1"]["brier"]
            for mv, desc in ABLATIONS:
                ev = (
                    evals[season]["integrated_v1"]
                    if mv == "integrated_v1"
                    else eval_model(session, season, mv)
                )
                contrib = ev["brier"] - full_brier
                report["ablation"].setdefault(str(season), {})[mv] = {
                    "brier": ev["brier"], "ece": ev["ece"], "contrib_vs_full": contrib,
                }
                contrib_str = "    —   " if mv == "integrated_v1" else f"{contrib:>+9.4f}"
                print(
                    f"  {season:>6} {desc:>38} {ev['brier']:>8.4f}"
                    f" {ev['ece']:>8.4f} {contrib_str}"
                )
            print()

    json_path = DATA_DIR / "validation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  JSON report saved to: {json_path}")

    # ------------------------------------------------------------------
    # 4. Honest verdict — best model per season vs the market.
    # ------------------------------------------------------------------
    print("\n" + _hr())
    print("[4] HONEST VERDICT\n")
    for season in args.seasons:
        bm_all = report["benchmark"].get(str(season), {})
        # Consensus Brier is identical across models (same games); grab one.
        cons_brier = None
        for mv in MODELS:
            b = bm_all.get(mv)
            if b and b.get("available"):
                cons_brier = b["consensus_brier"]
                break
        if cons_brier is None:
            print(f"  {season}: Squiggle tips unavailable — no market comparison.")
            continue

        # Best model by benchmark Brier (scored on the consensus game set).
        scored = [
            (mv, bm_all[mv]["our_brier"])
            for mv in MODELS
            if bm_all.get(mv) and bm_all[mv].get("available")
        ]
        best_mv, best_brier = min(scored, key=lambda x: x[1])
        delta = cons_brier - best_brier
        edge = "BEATS" if best_brier < cons_brier else ("TIES" if abs(delta) < 1e-4 else "loses to")
        ridge_b = bm_all.get("ridge_v1", {}).get("our_brier")
        log_b = bm_all.get("logistic_v1", {}).get("our_brier")
        print(
            f"  {season}: best model = {LABELS[best_mv]} "
            f"(Brier {best_brier:.4f}) {edge} consensus {cons_brier:.4f} "
            f"by {delta:+.4f}."
        )
        if ridge_b is not None and log_b is not None:
            print(
                f"          direct logistic {log_b:.4f} vs ridge->cdf {ridge_b:.4f}"
                f" ({log_b - ridge_b:+.4f})."
            )
        # Headline: how does the stacked integrated model fare vs the market?
        integ = bm_all.get("integrated_v1", {})
        if integ.get("available"):
            ib = integ["our_brier"]
            idelta = cons_brier - ib
            iverdict = (
                "BEATS" if ib < cons_brier
                else ("TIES" if abs(idelta) < 1e-4 else "loses to")
            )
            print(
                f"          integrated_v1 {ib:.4f} {iverdict} consensus {cons_brier:.4f}"
                f" by {idelta:+.4f}."
            )
    print(_hr())
    return 0


if __name__ == "__main__":
    sys.exit(main())
