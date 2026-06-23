"""Calibration / reliability tooling for AFL win-probability models.

Builds reliability tables (predicted vs observed win rate per decile),
computes Expected Calibration Error, and renders a static reliability-curve
PNG via matplotlib. Pure consumers of the walk-forward harness in
``src.eval.backtest`` — nothing here modifies the underlying models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from src.config import DATA_DIR
from src.eval.backtest import match_predictions
from src.eval.metrics import (
    brier_score,
    calibration_bins,
    expected_calibration_error,
    maximum_calibration_error,
)


@dataclass
class CalibrationResult:
    model_version: str
    season: int
    n_games: int
    ece: float
    mce: float
    brier: float
    bins: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "season": self.season,
            "n_games": self.n_games,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "bins": self.bins,
        }


def reliability_table(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Reliability bins with predicted mean, observed rate, and gap per bin."""
    bins = calibration_bins(np.asarray(probs, float), np.asarray(outcomes, float), n_bins)
    for b in bins:
        b["gap"] = b["actual_rate"] - b["predicted_mean"]
    return bins


def compute_calibration(
    session: Session,
    season: int,
    model_version: str,
    n_bins: int = 10,
) -> CalibrationResult:
    """Run the walk-forward harness for a model/season and summarise calibration."""
    preds = match_predictions(session, season, model_version)
    probs = np.array([p.home_win_prob for p in preds])
    outcomes = np.array([p.outcome for p in preds])
    return CalibrationResult(
        model_version=model_version,
        season=season,
        n_games=len(preds),
        ece=expected_calibration_error(probs, outcomes, n_bins=n_bins),
        mce=maximum_calibration_error(probs, outcomes, n_bins=n_bins),
        brier=brier_score(probs, outcomes) if len(probs) else 0.0,
        bins=reliability_table(probs, outcomes, n_bins=n_bins),
    )


def format_reliability_table(result: CalibrationResult) -> str:
    """Render a CalibrationResult's bins as a fixed-width text table."""
    lines = [
        f"  {'bin':>12}  {'pred':>6}  {'actual':>6}  {'gap':>7}  {'n':>4}",
        "  " + "-" * 44,
    ]
    for b in result.bins:
        lines.append(
            f"  {b['bin_low']:.1f}-{b['bin_high']:.1f}".rjust(14)
            + f"  {b['predicted_mean']:.3f}  {b['actual_rate']:.3f}"
            + f"  {b['gap']:+.3f}  {b['count']:>4}"
        )
    lines.append(
        f"  -> n={result.n_games}  ECE={result.ece:.4f}  "
        f"MCE={result.mce:.4f}  Brier={result.brier:.4f}"
    )
    return "\n".join(lines)


def plot_reliability(
    results: list[CalibrationResult],
    out_path: Path | str | None = None,
) -> Path:
    """Save a reliability-curve PNG comparing one or more CalibrationResults.

    Uses a non-interactive Agg backend so it works headless.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = DATA_DIR / "calibration_reliability.png"
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    markers = ["o", "s", "^", "D", "v", "P"]
    for i, res in enumerate(results):
        if not res.bins:
            continue
        xs = [b["predicted_mean"] for b in res.bins]
        ys = [b["actual_rate"] for b in res.bins]
        ax.plot(
            xs,
            ys,
            marker=markers[i % len(markers)],
            linewidth=1.5,
            label=(
                f"{res.model_version} {res.season} "
                f"(ECE={res.ece:.3f}, Brier={res.brier:.3f})"
            ),
        )

    ax.set_xlabel("Predicted home-win probability")
    ax.set_ylabel("Observed home-win rate")
    ax.set_title("Reliability curve — AFL win-probability models")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
