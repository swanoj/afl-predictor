"""Walk-forward backtesting harness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import TRAIN_SEASONS, VALIDATION_SEASON
from src.db.models import BacktestResult, Match
from src.eval.metrics import brier_score, log_loss, mae
from src.macro.calibration_model import ProbabilityCalibrator
from src.macro.elo import EloEngine, fit_margin_scale
from src.macro.prob_model import TeamWinProbLogistic, build_win_prob_model
from src.macro.team_model import (
    INTEGRATED_FEATURE_COLUMNS,
    TeamMarginModel,
    build_full_feature_frame,
    build_integrated_feature_frame,
)


@dataclass
class BacktestReport:
    model_version: str
    season: int
    brier: float
    mae_margin: float
    log_loss_val: float
    n_games: int
    home_win_accuracy: float

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "season": self.season,
            "brier": self.brier,
            "mae_margin": self.mae_margin,
            "log_loss": self.log_loss_val,
            "n_games": self.n_games,
            "home_win_accuracy": self.home_win_accuracy,
        }


@dataclass
class MatchPrediction:
    """Per-match prediction emitted by the walk-forward harness."""

    squiggle_id: int
    year: int
    round: int
    home_team: str
    away_team: str
    home_win_prob: float
    outcome: float  # 1.0 if home won
    margin_pred: float
    margin_actual: float


def elo_match_predictions(session: Session, season: int) -> list[MatchPrediction]:
    """Walk-forward Elo predictions, one row per completed match in *season*."""
    train_matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year < season)  # noqa: E712
        .order_by(Match.year, Match.round)
    ).all()

    train_seasons = sorted({m.year for m in train_matches})
    margin_scale = fit_margin_scale(session, train_seasons)

    elo = EloEngine()
    for m in train_matches:
        if m.home_score is None or m.away_score is None:
            continue
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    test_matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year == season)  # noqa: E712
        .order_by(Match.round)
    ).all()

    preds: list[MatchPrediction] = []
    for m in test_matches:
        if m.home_score is None or m.away_score is None:
            continue
        pred = elo.predict(m.home_team, m.away_team)
        preds.append(
            MatchPrediction(
                squiggle_id=m.squiggle_id,
                year=m.year,
                round=m.round,
                home_team=m.home_team,
                away_team=m.away_team,
                home_win_prob=pred["home_win_prob"],
                outcome=1.0 if m.home_score > m.away_score else 0.0,
                margin_pred=pred["elo_diff"] * margin_scale,
                margin_actual=m.home_score - m.away_score,
            )
        )
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)
    return preds


def ridge_match_predictions(session: Session, season: int) -> list[MatchPrediction]:
    """Walk-forward Ridge predictions, one row per completed match in *season*."""
    train_seasons = [s for s in TRAIN_SEASONS if s < season]
    if not train_seasons:
        train_seasons = list(range(season - 5, season))

    ridge = TeamMarginModel()
    ridge.fit(session, train_seasons)

    elo = EloEngine()
    pre_matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year < season)  # noqa: E712
        .order_by(Match.year, Match.round)
    ).all()
    for m in pre_matches:
        if m.home_score is None or m.away_score is None:
            continue
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    test_matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year == season)  # noqa: E712
        .order_by(Match.round)
    ).all()

    preds: list[MatchPrediction] = []
    for m in test_matches:
        if m.home_score is None or m.away_score is None:
            continue
        margin = ridge.predict_margin(session, m, elo)
        preds.append(
            MatchPrediction(
                squiggle_id=m.squiggle_id,
                year=m.year,
                round=m.round,
                home_team=m.home_team,
                away_team=m.away_team,
                home_win_prob=ridge.predict_win_prob(margin),
                outcome=1.0 if m.home_score > m.away_score else 0.0,
                margin_pred=margin,
                margin_actual=m.home_score - m.away_score,
            )
        )
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)
    return preds


# ----------------------------------------------------------------------------
# Direct win-probability models (logistic / GBM) + calibration, all driven off
# the shared walk-forward feature frame (sliced by season — leakage-free).
# ----------------------------------------------------------------------------


def _train_seasons_before(frame, season: int) -> list[int]:
    return sorted(int(y) for y in frame["year"].unique() if int(y) < season)


def _slice(frame, seasons: list[int]):
    return frame[frame["year"].isin(seasons)]


def _preds_from_rows(rows, probs: np.ndarray) -> list[MatchPrediction]:
    """Build MatchPrediction rows for win-prob models.

    These models predict probability directly, not a margin, so ``margin_pred``
    is NaN (MAE is only meaningful for the margin models).
    """
    preds: list[MatchPrediction] = []
    for (_, r), p in zip(rows.iterrows(), probs):
        preds.append(
            MatchPrediction(
                squiggle_id=int(r["squiggle_id"]),
                year=int(r["year"]),
                round=int(r["round"]),
                home_team=r["home_team"],
                away_team=r["away_team"],
                home_win_prob=float(p),
                outcome=float(r["home_win"]),
                margin_pred=float("nan"),
                margin_actual=float(r["margin"]),
            )
        )
    return preds


def winprob_match_predictions(
    session: Session, season: int, kind: str
) -> list[MatchPrediction]:
    """Walk-forward predictions for a direct win-prob model (``logistic``/``gbm``)."""
    frame = build_full_feature_frame(session)
    train_seasons = _train_seasons_before(frame, season)
    if not train_seasons:
        raise ValueError(f"No training seasons available before {season}")

    model = build_win_prob_model(kind)
    train = _slice(frame, train_seasons)
    model.fit_frame(train, train["home_win"].to_numpy())

    test = _slice(frame, [season])
    probs = model.predict_proba_frame(test)
    return _preds_from_rows(test, probs)


def fit_calibrated_winprob(
    session: Session,
    season: int,
    kind: str,
    method: str,
) -> tuple[object, ProbabilityCalibrator]:
    """Fit a leakage-free calibrated win-prob model for predicting *season*.

    Returns ``(base_model, calibrator)`` where ``base_model`` is fit on *all*
    seasons strictly before ``season`` and ``calibrator`` is fit on a held-out
    pre-test slice (never the target season). This is the shared, leak-free
    training recipe used by both the walk-forward backtest and the production
    prediction service:

      1. fit base model on ``train_seasons[:-1]``;
      2. predict the held-out season ``train_seasons[-1]`` and fit the
         calibrator on (raw_prob, outcome) there;
      3. refit the base model on *all* ``train_seasons``.

    The caller obtains predictions by computing the model's raw probability for
    any match in (or after) ``season`` and passing it through ``calibrator``.
    """
    frame = build_full_feature_frame(session)
    train_seasons = _train_seasons_before(frame, season)
    if len(train_seasons) < 2:
        raise ValueError(
            f"Need >=2 training seasons before {season} to fit a calibrator"
        )

    holdout = train_seasons[-1]
    base_train = train_seasons[:-1]

    base = build_win_prob_model(kind)
    bt = _slice(frame, base_train)
    base.fit_frame(bt, bt["home_win"].to_numpy())

    hold = _slice(frame, [holdout])
    raw_hold = base.predict_proba_frame(hold)
    calibrator = ProbabilityCalibrator(method=method).fit(
        raw_hold, hold["home_win"].to_numpy()
    )

    base_full = build_win_prob_model(kind)
    tr = _slice(frame, train_seasons)
    base_full.fit_frame(tr, tr["home_win"].to_numpy())

    return base_full, calibrator


def calibrated_match_predictions(
    session: Session,
    season: int,
    kind: str,
    method: str,
) -> list[MatchPrediction]:
    """Walk-forward predictions for a calibrated win-prob model.

    Calibration is fit on a *held-out* slice of training data (the most recent
    pre-test season), never on the test season, so there is no leakage. The
    base model + calibrator are produced by :func:`fit_calibrated_winprob`; we
    then predict the test season and apply the fitted calibrator.
    """
    base_full, calibrator = fit_calibrated_winprob(session, season, kind, method)

    frame = build_full_feature_frame(session)
    test = _slice(frame, [season])
    raw = base_full.predict_proba_frame(test)
    probs = calibrator.transform(raw)
    return _preds_from_rows(test, probs)


# ----------------------------------------------------------------------------
# Integrated (stacked) model: upgraded-Elo rating source + enriched features
# (avail_diff, xscore_diff) + direct logistic + post-hoc calibration. Built off
# the dedicated walk-forward integrated frame (sliced by season -> leak-free).
# The same code path powers the feature ablations.
# ----------------------------------------------------------------------------

# Default calibration method for the integrated model. Sigmoid (Platt) was the
# best-calibrated base on 2024 Brier in the validation report.
INTEGRATED_CALIB_METHOD = "sigmoid"


def integrated_match_predictions(
    session: Session,
    season: int,
    *,
    feature_columns: list[str] | None = None,
    method: str = INTEGRATED_CALIB_METHOD,
    use_flat_elo: bool = False,
) -> list[MatchPrediction]:
    """Walk-forward predictions for the integrated model (and its ablations).

    Pipeline: fit logistic on the enriched features, fit a calibrator on a
    held-out pre-test season, refit on all training seasons, then predict and
    recalibrate the test season — exactly mirroring ``calibrated_match_predictions``
    so calibration never sees the test season.

    Ablation knobs (used by the validation report):
      * ``feature_columns`` — drop a feature (e.g. without ``avail_diff``).
      * ``use_flat_elo``    — swap the upgraded rating source for the flat one.
    """
    cols = list(feature_columns) if feature_columns is not None else list(
        INTEGRATED_FEATURE_COLUMNS
    )

    frame = build_integrated_feature_frame(session)
    if use_flat_elo:
        # Substitute the flat-HGA Elo diff for the upgraded one in-place on a
        # copy, keeping the column name so the model spec is identical.
        frame = frame.copy()
        frame["elo_diff"] = frame["elo_diff_flat"]

    train_seasons = _train_seasons_before(frame, season)
    if len(train_seasons) < 2:
        raise ValueError(
            f"Need >=2 training seasons before {season} to fit the integrated calibrator"
        )

    holdout = train_seasons[-1]
    base_train = train_seasons[:-1]

    base = TeamWinProbLogistic(feature_columns=cols)
    bt = _slice(frame, base_train)
    base.fit_frame(bt, bt["home_win"].to_numpy())

    hold = _slice(frame, [holdout])
    raw_hold = base.predict_proba_frame(hold)
    calibrator = ProbabilityCalibrator(method=method).fit(
        raw_hold, hold["home_win"].to_numpy()
    )

    base_full = TeamWinProbLogistic(feature_columns=cols)
    tr = _slice(frame, train_seasons)
    base_full.fit_frame(tr, tr["home_win"].to_numpy())

    test = _slice(frame, [season])
    raw = base_full.predict_proba_frame(test)
    probs = calibrator.transform(raw)
    return _preds_from_rows(test, probs)


def _integrated_ablation(session: Session, season: int, which: str) -> list[MatchPrediction]:
    """Dispatch an integrated ablation variant by name."""
    if which == "drop_avail":
        cols = [c for c in INTEGRATED_FEATURE_COLUMNS if c != "avail_diff"]
        return integrated_match_predictions(session, season, feature_columns=cols)
    if which == "drop_xscore":
        cols = [c for c in INTEGRATED_FEATURE_COLUMNS if c != "xscore_diff"]
        return integrated_match_predictions(session, season, feature_columns=cols)
    if which == "flat_elo":
        return integrated_match_predictions(session, season, use_flat_elo=True)
    raise ValueError(f"Unknown integrated ablation: {which}")


def match_predictions(
    session: Session, season: int, model_version: str
) -> list[MatchPrediction]:
    """Dispatch to the per-match walk-forward harness for a model version.

    Recognised versions: ``elo_v1``, ``ridge_v1``, ``logistic_v1``, ``gbm_v1``,
    and calibrated models encoded as ``calibrated:{logistic|gbm}:{isotonic|sigmoid}``.
    """
    if model_version == "elo_v1":
        return elo_match_predictions(session, season)
    if model_version == "ridge_v1":
        return ridge_match_predictions(session, season)
    if model_version == "logistic_v1":
        return winprob_match_predictions(session, season, "logistic")
    if model_version == "gbm_v1":
        return winprob_match_predictions(session, season, "gbm")
    if model_version.startswith("calibrated:"):
        _, kind, method = model_version.split(":")
        return calibrated_match_predictions(session, season, kind, method)
    if model_version == "integrated_v1":
        return integrated_match_predictions(session, season)
    if model_version.startswith("integrated:"):
        _, which = model_version.split(":", 1)
        return _integrated_ablation(session, season, which)
    raise ValueError(f"Unknown model_version: {model_version}")


def _report_from_preds(
    model_version: str, season: int, preds: list[MatchPrediction]
) -> BacktestReport:
    """Summarise a list of MatchPredictions into a BacktestReport."""
    probs = np.array([p.home_win_prob for p in preds])
    outcomes = np.array([p.outcome for p in preds])
    margin_preds = np.array([p.margin_pred for p in preds])
    margin_actuals = np.array([p.margin_actual for p in preds])
    return BacktestReport(
        model_version=model_version,
        season=season,
        brier=brier_score(probs, outcomes),
        mae_margin=mae(margin_preds, margin_actuals),
        log_loss_val=log_loss(probs, outcomes),
        n_games=len(preds),
        home_win_accuracy=float(np.mean((probs >= 0.5) == outcomes)),
    )


def walk_forward_model(
    session: Session, season: int, model_version: str
) -> BacktestReport:
    """Generic walk-forward backtest for any registered model version."""
    preds = match_predictions(session, season, model_version)
    return _report_from_preds(model_version, season, preds)


def walk_forward_elo(session: Session, season: int) -> BacktestReport:
    """Backtest Elo predictions on a season using walk-forward updates."""
    train_matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year < season)  # noqa: E712
        .order_by(Match.year, Match.round)
    ).all()

    train_seasons = sorted({m.year for m in train_matches})
    margin_scale = fit_margin_scale(session, train_seasons)

    elo = EloEngine()
    for m in train_matches:
        if m.home_score is None or m.away_score is None:
            continue
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    test_matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year == season)  # noqa: E712
        .order_by(Match.round)
    ).all()

    probs = []
    outcomes = []
    margin_preds = []
    margin_actuals = []

    for m in test_matches:
        if m.home_score is None or m.away_score is None:
            continue
        pred = elo.predict(m.home_team, m.away_team)
        probs.append(pred["home_win_prob"])
        outcomes.append(1.0 if m.home_score > m.away_score else 0.0)
        margin_preds.append(pred["elo_diff"] * margin_scale)
        margin_actuals.append(m.home_score - m.away_score)
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    probs_arr = np.array(probs)
    outcomes_arr = np.array(outcomes)

    return BacktestReport(
        model_version="elo_v1",
        season=season,
        brier=brier_score(probs_arr, outcomes_arr),
        mae_margin=mae(np.array(margin_preds), np.array(margin_actuals)),
        log_loss_val=log_loss(probs_arr, outcomes_arr),
        n_games=len(probs),
        home_win_accuracy=float(np.mean((probs_arr >= 0.5) == outcomes_arr)),
    )


def walk_forward_ridge(session: Session, season: int) -> BacktestReport:
    """Backtest Ridge margin model with walk-forward Elo features."""
    train_seasons = [s for s in TRAIN_SEASONS if s < season]
    if not train_seasons:
        train_seasons = list(range(season - 5, season))

    ridge = TeamMarginModel()
    ridge.fit(session, train_seasons)

    elo = EloEngine()
    pre_matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year < season)  # noqa: E712
        .order_by(Match.year, Match.round)
    ).all()
    for m in pre_matches:
        if m.home_score is None or m.away_score is None:
            continue
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    test_matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year == season)  # noqa: E712
        .order_by(Match.round)
    ).all()

    probs = []
    outcomes = []
    margin_preds = []
    margin_actuals = []

    for m in test_matches:
        if m.home_score is None or m.away_score is None:
            continue
        margin = ridge.predict_margin(session, m, elo)
        # Independent probability from ridge's own predicted margin (not Elo's).
        probs.append(ridge.predict_win_prob(margin))
        margin_preds.append(margin)
        margin_actuals.append(m.home_score - m.away_score)
        outcomes.append(1.0 if m.home_score > m.away_score else 0.0)
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    probs_arr = np.array(probs)
    outcomes_arr = np.array(outcomes)

    return BacktestReport(
        model_version="ridge_v1",
        season=season,
        brier=brier_score(probs_arr, outcomes_arr),
        mae_margin=mae(np.array(margin_preds), np.array(margin_actuals)),
        log_loss_val=log_loss(probs_arr, outcomes_arr),
        n_games=len(probs),
        home_win_accuracy=float(np.mean((probs_arr >= 0.5) == outcomes_arr)),
    )


def save_backtest_result(session: Session, report: BacktestReport) -> None:
    session.add(
        BacktestResult(
            model_version=report.model_version,
            season=report.season,
            brier=report.brier,
            mae_margin=report.mae_margin,
            log_loss=report.log_loss_val,
            n_games=report.n_games,
            created_at=datetime.utcnow(),
        )
    )


def run_all_backtests(session: Session, season: int = VALIDATION_SEASON) -> list[BacktestReport]:
    reports = [
        walk_forward_elo(session, season),
        walk_forward_ridge(session, season),
    ]
    for r in reports:
        save_backtest_result(session, r)
    return reports
