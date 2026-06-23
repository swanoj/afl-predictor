"""Pure-Python lineup availability adjustments for production serving.

No numpy, pandas, scikit-learn, or SQLAlchemy — safe to import on Render where
the runtime image excludes the heavy modelling stack. Offline training code in
``src.ingest.lineups`` mirrors the same maths against the full engine DB.

Typical flow at request time::

    values = load_values_from_db(session)          # intelligence/lineups
    lineups = derive_match_lineups(...)            # intelligence/lineups
    margin_adj = home_away_margin_adjustment(...)  # this module
    home_p, away_p = shift_win_prob(base_home, margin_adj)
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Fitted scalar from scripts/eval_availability.py (fantasy points -> margin pts).
DEFAULT_POINTS_PER_VALUE = 0.024

# Linear win-prob nudge per margin point (logistic-ish, kept deliberately simple).
DEFAULT_WIN_PROB_SENSITIVITY = 0.02

PROB_FLOOR = 0.05
PROB_CEIL = 0.95

REPLACEMENT_PERCENTILE = 25.0

# Type alias: (player_name, team) -> shrunk fantasy-point value.
PlayerValueMap = Mapping[tuple[str, str], float]


def _lookup_value(
    values: PlayerValueMap,
    player_name: str,
    team: str,
    replacement: float,
) -> float:
    return float(values.get((player_name, team), replacement))


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def replacement_level(
    values: PlayerValueMap,
    *,
    min_games_sample: int | None = None,
) -> float:
    """Replacement fantasy-point level for unknown / debutant players.

    When ``min_games_sample`` is omitted every value in the map is used (the
    serving export is already shrunk). When provided, only entries whose
    companion metadata met the sample threshold should be pre-filtered by the
    caller — this helper only sees floats.
    """
    pool = list(values.values())
    if not pool:
        return 0.0
    _ = min_games_sample  # reserved for callers that pre-filter the map
    return _percentile(pool, REPLACEMENT_PERCENTILE)


def lineup_value(
    player_names: Sequence[str],
    team: str,
    values: PlayerValueMap,
    replacement: float | None = None,
) -> float:
    """Sum of player values for a named lineup (unknown players -> replacement)."""
    rep = replacement if replacement is not None else replacement_level(values)
    return float(
        sum(_lookup_value(values, name, team, rep) for name in player_names)
    )


def team_availability_margin(
    lineup: Sequence[str],
    team: str,
    baseline_lineup_value: float | None,
    values: PlayerValueMap,
    *,
    replacement: float | None = None,
    points_per_value: float = DEFAULT_POINTS_PER_VALUE,
) -> float:
    """Margin-points adjustment for one team vs its baseline-strength lineup.

    Positive -> stronger than baseline (stars returning); negative -> key outs.
    Returns ``0.0`` when baseline is unknown or the lineup is empty.
    """
    if not lineup or baseline_lineup_value is None:
        return 0.0
    rep = replacement if replacement is not None else replacement_level(values)
    current = lineup_value(lineup, team, values, rep)
    return (current - baseline_lineup_value) * points_per_value


def home_away_margin_adjustment(
    home_lineup: Sequence[str],
    away_lineup: Sequence[str],
    home_team: str,
    away_team: str,
    home_baseline: float | None,
    away_baseline: float | None,
    values: PlayerValueMap,
    *,
    replacement: float | None = None,
    points_per_value: float = DEFAULT_POINTS_PER_VALUE,
) -> float:
    """Net margin adjustment (home minus away) from lineup availability."""
    home_adj = team_availability_margin(
        home_lineup,
        home_team,
        home_baseline,
        values,
        replacement=replacement,
        points_per_value=points_per_value,
    )
    away_adj = team_availability_margin(
        away_lineup,
        away_team,
        away_baseline,
        values,
        replacement=replacement,
        points_per_value=points_per_value,
    )
    return home_adj - away_adj


def shift_win_prob(
    base_home_prob: float,
    margin_adj: float,
    *,
    sensitivity: float = DEFAULT_WIN_PROB_SENSITIVITY,
    prob_floor: float = PROB_FLOOR,
    prob_ceil: float = PROB_CEIL,
) -> tuple[float, float]:
    """Apply a linear lineup nudge to home win probability.

    ``base_home_prob`` may be ``0``–``1`` or ``0``–``100`` (detected when
    ``base_home_prob > 1``). Returns ``(home_prob, away_prob)`` in the same
    scale as the input.

    Logistic-ish rule: ``home += sensitivity * margin_adj``, then clamp.
    """
    pct_scale = base_home_prob > 1.0
    lo = prob_floor * 100 if pct_scale else prob_floor
    hi = prob_ceil * 100 if pct_scale else prob_ceil
    home = base_home_prob + sensitivity * margin_adj
    home = max(lo, min(hi, home))
    if pct_scale:
        away = 100.0 - home
    else:
        away = 1.0 - home
    return home, away


def apply_lineup_adjustment(
    base_home_prob: float,
    home_lineup: Sequence[str],
    away_lineup: Sequence[str],
    home_team: str,
    away_team: str,
    home_baseline: float | None,
    away_baseline: float | None,
    values: PlayerValueMap,
    *,
    replacement: float | None = None,
    points_per_value: float = DEFAULT_POINTS_PER_VALUE,
    sensitivity: float = DEFAULT_WIN_PROB_SENSITIVITY,
) -> dict[str, float | str]:
    """One-shot: margin adjustment + win-prob shift with provenance fields."""
    margin_adj = home_away_margin_adjustment(
        home_lineup,
        away_lineup,
        home_team,
        away_team,
        home_baseline,
        away_baseline,
        values,
        replacement=replacement,
        points_per_value=points_per_value,
    )
    home_wp, away_wp = shift_win_prob(
        base_home_prob, margin_adj, sensitivity=sensitivity
    )
    return {
        "home_win_prob": home_wp,
        "away_win_prob": away_wp,
        "lineup_margin_adj": margin_adj,
        "win_prob_source": "logistic+sigmoid+lineup",
    }
