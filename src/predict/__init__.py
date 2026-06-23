"""Unified match-prediction service.

The headline win probability comes from a logistic-regression model passed
through a sigmoid (Platt) calibrator (the best-calibrated model per the Phase
B/C backtests). Margin & scores come from the Ridge ``TeamMarginModel``. The
Monte Carlo ``HybridSimulator`` is used *only* to produce player-level
projections; its own win-probability / score outputs are intentionally ignored
here.

Import policy
-------------
The lightweight (de)serialization helpers live in :mod:`src.predict.serialize`
and import only SQLAlchemy, so they are re-exported eagerly here. The heavy
modelling entry points (``predict_match`` / ``predict_round`` /
``predict_season`` / ``clear_cache``) live in :mod:`src.predict.service`, which
pulls in numpy/pandas/scikit-learn/numba. Those are exposed **lazily** via
module ``__getattr__`` (PEP 562) so that simply importing ``src.predict`` (or
``src.predict.serialize``) does NOT drag in the modelling stack — critical for
the lean serving path, where everything is precomputed.
"""

from typing import TYPE_CHECKING

from src.predict.serialize import stored_to_item, upsert_stored_prediction

_LAZY = {"predict_match", "predict_round", "predict_season", "clear_cache"}

__all__ = [
    "predict_match",
    "predict_round",
    "predict_season",
    "upsert_stored_prediction",
    "stored_to_item",
    "clear_cache",
]

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from src.predict.service import (
        clear_cache,
        predict_match,
        predict_round,
        predict_season,
    )


def __getattr__(name: str):
    if name in _LAZY:
        from src.predict import service

        return getattr(service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
