"""Direct win-probability models for AFL macro predictions.

Where :class:`~src.macro.team_model.TeamMarginModel` predicts a margin and
converts it to a win probability via ``norm.cdf(margin / sigma)``, the models
here predict the binary home-win outcome *directly* and therefore optimise the
log-loss / Brier objective that we actually care about.

Two estimators are provided, trained on the identical feature set built by
``team_model.compute_match_features`` (``FEATURE_COLUMNS``):

* :class:`TeamWinProbLogistic` — L2 logistic regression on standardised
  features (a GLM on the binary target).
* :class:`TeamWinProbGBM` — sklearn ``HistGradientBoostingClassifier`` to
  capture any non-linear interactions. We deliberately use sklearn's
  histogram booster rather than adding a LightGBM dependency.

Both expose ``fit_frame`` / ``predict_proba_frame`` (operating on a prepared
feature DataFrame, used by the walk-forward harness for speed) plus
``fit`` / ``predict_win_prob`` convenience methods that build features from the
database for a single match. All training is leakage-free as long as the caller
only fits on rows from seasons strictly before the season being predicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from src.macro.elo import EloEngine
from src.macro.team_model import FEATURE_COLUMNS, compute_match_features

# Probabilities are clipped away from {0, 1} so log-loss stays finite and a
# single freak result can't dominate the score.
PROB_EPS = 1e-4


def _clip(probs: np.ndarray) -> np.ndarray:
    return np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)


@dataclass
class TeamWinProbLogistic:
    """L2-regularised logistic regression on the macro feature set."""

    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    C: float = 1.0
    model: Pipeline | None = None
    fitted: bool = False

    def _new_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs"),
                ),
            ]
        )

    def fit_frame(self, X: pd.DataFrame, y: np.ndarray) -> "TeamWinProbLogistic":
        self.model = self._new_pipeline()
        self.model.fit(X[self.feature_columns].to_numpy(dtype=float), np.asarray(y, dtype=float))
        self.fitted = True
        return self

    def predict_proba_frame(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise RuntimeError("TeamWinProbLogistic is not fitted")
        proba = self.model.predict_proba(X[self.feature_columns].to_numpy(dtype=float))[:, 1]
        return _clip(proba)

    def fit(self, session: Session, seasons: list[int]) -> "TeamWinProbLogistic":
        from src.macro.team_model import build_full_feature_frame

        frame = build_full_feature_frame(session)
        train = frame[frame["year"].isin(seasons)]
        if len(train) < 50:
            raise ValueError("Insufficient training data for logistic win-prob model")
        return self.fit_frame(train, train["home_win"].to_numpy())

    def predict_win_prob(self, session: Session, match, elo_engine: EloEngine) -> float:
        feats = compute_match_features(session, match, elo_engine)
        X = pd.DataFrame([feats])
        return float(self.predict_proba_frame(X)[0])


@dataclass
class TeamWinProbGBM:
    """Histogram gradient-boosting classifier on the macro feature set.

    Hyper-parameters are conservative because the training set is small
    (~1k–1.3k games); strong L2 regularisation and a shallow tree depth guard
    against overfitting the binary target.
    """

    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    learning_rate: float = 0.05
    max_iter: int = 300
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    random_state: int = 42
    model: HistGradientBoostingClassifier | None = None
    fitted: bool = False

    def _new_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=self.random_state,
        )

    def fit_frame(self, X: pd.DataFrame, y: np.ndarray) -> "TeamWinProbGBM":
        self.model = self._new_model()
        self.model.fit(X[self.feature_columns].to_numpy(dtype=float), np.asarray(y, dtype=int))
        self.fitted = True
        return self

    def predict_proba_frame(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted or self.model is None:
            raise RuntimeError("TeamWinProbGBM is not fitted")
        proba = self.model.predict_proba(X[self.feature_columns].to_numpy(dtype=float))[:, 1]
        return _clip(proba)

    def fit(self, session: Session, seasons: list[int]) -> "TeamWinProbGBM":
        from src.macro.team_model import build_full_feature_frame

        frame = build_full_feature_frame(session)
        train = frame[frame["year"].isin(seasons)]
        if len(train) < 50:
            raise ValueError("Insufficient training data for GBM win-prob model")
        return self.fit_frame(train, train["home_win"].to_numpy())

    def predict_win_prob(self, session: Session, match, elo_engine: EloEngine) -> float:
        feats = compute_match_features(session, match, elo_engine)
        X = pd.DataFrame([feats])
        return float(self.predict_proba_frame(X)[0])


def build_win_prob_model(kind: str, **kwargs):
    """Factory: ``"logistic"`` -> logistic GLM, ``"gbm"`` -> HistGB classifier."""
    if kind == "logistic":
        return TeamWinProbLogistic(**kwargs)
    if kind == "gbm":
        return TeamWinProbGBM(**kwargs)
    raise ValueError(f"Unknown win-prob model kind: {kind!r}")
