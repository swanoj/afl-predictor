"""Standalone eval: does lineup value add signal beyond team rating?

Hypothesis
----------
"The summed player-value of a team's lineup, relative to that team's recent
baseline lineup, predicts match margin BEYOND the team's Elo rating."

We test it honestly with a strict chronological walk-forward (every quantity
for a match uses only games played *before* it):

  1. Pearson correlation of the availability differential with margin.
  2. Partial signal: correlate availability with the margin residual left over
     after Elo (the only fair "beyond team rating" test).
  3. Incremental ridge (margin MAE) and logistic (win Brier) on a 2024-2025
     hold-out: [elo] vs [elo + availability].
  4. The fitted fantasy-points -> margin-points scalar (feeds
     ``player_value.DEFAULT_POINTS_PER_VALUE``).
  5. Approach (b): ridge on per-player presence indicators, compared to the
     fantasy-score baseline (approach a).

OPTIMISM CAVEAT: historical lineups are the *realized* 22 (who actually
played), so these numbers are an UPPER BOUND on the signal a real pre-game
forecaster (working from the named 22) would capture. See
``src/ingest/lineups.py`` for the full discussion.

Usage:
    python scripts/eval_availability.py
"""

from __future__ import annotations

import sys
from collections import deque, namedtuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sqlalchemy import select

from src.db.models import Match, PlayerGameLog
from src.db.session import get_session, init_db
from src.eval.metrics import brier_score, mae
from src.macro.elo import EloEngine
from src.micro.player_value import (
    DEFAULT_MIN_GAMES,
    DEFAULT_SHRINKAGE,
    REPLACEMENT_PERCENTILE,
    fantasy_points,
    shrink,
)

TRAIN_SEASONS = list(range(2018, 2024))
TEST_SEASONS = [2024, 2025]
BASELINE_WINDOW = 12


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
MatchRow = namedtuple(
    "MatchRow", "id year round home_team away_team home_score away_score"
)


def _load_matches(session):
    """Materialize completed matches as plain tuples (detached-safe)."""
    matches = session.scalars(
        select(Match)
        .where(Match.complete == True)  # noqa: E712
        .order_by(Match.year, Match.round, Match.date, Match.id)
    ).all()
    return [
        MatchRow(m.id, m.year, m.round, m.home_team, m.away_team, m.home_score, m.away_score)
        for m in matches
        if m.home_score is not None and m.away_score is not None
    ]


def _load_lineups(session) -> dict[int, dict[str, list]]:
    """match_id -> {team -> [(player_name, fantasy_points), ...]}."""
    logs = session.scalars(select(PlayerGameLog)).all()
    out: dict[int, dict[str, list]] = {}
    for log in logs:
        out.setdefault(log.match_id, {}).setdefault(log.team, []).append(
            (log.player_name, fantasy_points(log))
        )
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), pct))


# ---------------------------------------------------------------------------
# Approach (a): walk-forward fantasy-score values, baseline-relative
# ---------------------------------------------------------------------------
def build_features_fantasy(matches, lineups):
    """Strict walk-forward feature build.

    Returns list of dicts with: year, elo_diff, avail_diff (fantasy points),
    margin, home_win, has_baseline.
    """
    elo = EloEngine()
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    base: dict[str, deque] = {}

    def current_replacement() -> float:
        established = [
            sums[k] / counts[k] for k in sums if counts[k] >= max(DEFAULT_MIN_GAMES, 5)
        ]
        return _percentile(established, REPLACEMENT_PERCENTILE) if established else 0.0

    def player_value(name: str, team: str, replacement: float) -> float:
        key = (name, team)
        n = counts.get(key, 0)
        if n == 0:
            return replacement
        return shrink(sums[key] / n, n, replacement, DEFAULT_SHRINKAGE)

    rows = []
    for m in matches:
        ml = lineups.get(m.id, {})
        home_players = ml.get(m.home_team, [])
        away_players = ml.get(m.away_team, [])
        replacement = current_replacement()

        home_val = sum(player_value(n, m.home_team, replacement) for n, _ in home_players)
        away_val = sum(player_value(n, m.away_team, replacement) for n, _ in away_players)

        home_base = base.get(m.home_team)
        away_base = base.get(m.away_team)
        has_base = bool(home_base) and bool(away_base)
        if has_base:
            hb = sum(home_base) / len(home_base)
            ab = sum(away_base) / len(away_base)
            avail_diff = (home_val - hb) - (away_val - ab)
        else:
            avail_diff = 0.0

        pred = elo.predict(m.home_team, m.away_team)
        margin = m.home_score - m.away_score
        rows.append(
            {
                "year": m.year,
                "elo_diff": pred["elo_diff"],
                "avail_diff": avail_diff,
                "margin": margin,
                "home_win": 1.0 if margin > 0 else 0.0,
                "has_baseline": has_base,
            }
        )

        # ---- update walk-forward state AFTER recording the row ----
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)
        for name, fp in home_players:
            k = (name, m.home_team)
            sums[k] = sums.get(k, 0.0) + fp
            counts[k] = counts.get(k, 0) + 1
        for name, fp in away_players:
            k = (name, m.away_team)
            sums[k] = sums.get(k, 0.0) + fp
            counts[k] = counts.get(k, 0) + 1
        if home_players:
            base.setdefault(m.home_team, deque(maxlen=BASELINE_WINDOW)).append(home_val)
        if away_players:
            base.setdefault(m.away_team, deque(maxlen=BASELINE_WINDOW)).append(away_val)

    return rows


# ---------------------------------------------------------------------------
# Approach (b): ridge on per-player presence indicators
# ---------------------------------------------------------------------------
def fit_player_ridge(matches, lineups, train_seasons, alpha: float = 50.0):
    """Regress margin on (home presence - away presence) player indicators
    using only ``train_seasons``. Returns {(name, team): margin_coef}."""
    train = [m for m in matches if m.year in train_seasons]
    index: dict[tuple[str, str], int] = {}
    for m in train:
        ml = lineups.get(m.id, {})
        for name, _ in ml.get(m.home_team, []):
            index.setdefault((name, m.home_team), len(index))
        for name, _ in ml.get(m.away_team, []):
            index.setdefault((name, m.away_team), len(index))
    if not index:
        return {}

    X = np.zeros((len(train), len(index)), dtype=float)
    y = np.zeros(len(train), dtype=float)
    for i, m in enumerate(train):
        ml = lineups.get(m.id, {})
        for name, _ in ml.get(m.home_team, []):
            X[i, index[(name, m.home_team)]] += 1.0
        for name, _ in ml.get(m.away_team, []):
            X[i, index[(name, m.away_team)]] -= 1.0
        y[i] = m.home_score - m.away_score

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    return {key: float(model.coef_[idx]) for key, idx in index.items()}


def build_features_ridge_values(matches, lineups, coef):
    """Walk-forward, baseline-relative availability using STATIC ridge player
    coefficients (trained on past seasons only). Returns rows like approach a."""
    elo = EloEngine()
    base: dict[str, deque] = {}

    def lineup_val(players, team):
        return sum(coef.get((n, team), 0.0) for n, _ in players)

    rows = []
    for m in matches:
        ml = lineups.get(m.id, {})
        home_players = ml.get(m.home_team, [])
        away_players = ml.get(m.away_team, [])
        home_val = lineup_val(home_players, m.home_team)
        away_val = lineup_val(away_players, m.away_team)

        home_base = base.get(m.home_team)
        away_base = base.get(m.away_team)
        has_base = bool(home_base) and bool(away_base)
        if has_base:
            hb = sum(home_base) / len(home_base)
            ab = sum(away_base) / len(away_base)
            avail_diff = (home_val - hb) - (away_val - ab)
        else:
            avail_diff = 0.0

        margin = m.home_score - m.away_score
        rows.append(
            {
                "year": m.year,
                "avail_diff": avail_diff,
                "margin": margin,
                "has_baseline": has_base,
            }
        )
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)
        if home_players:
            base.setdefault(m.home_team, deque(maxlen=BASELINE_WINDOW)).append(home_val)
        if away_players:
            base.setdefault(m.away_team, deque(maxlen=BASELINE_WINDOW)).append(away_val)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _split(rows, seasons, keys):
    mask = [r for r in rows if r["year"] in seasons and r["has_baseline"]]
    return {k: np.asarray([r[k] for r in mask], dtype=float) for k in keys}


def _hr(c="="):
    return c * 72


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    init_db()
    with get_session() as session:
        matches = _load_matches(session)
        lineups = _load_lineups(session)

    print(_hr())
    print("AVAILABILITY EVAL — does lineup value beat team rating?")
    print(_hr())
    print(f"  matches (complete, scored): {len(matches)}")
    print(f"  train seasons: {TRAIN_SEASONS}   test seasons: {TEST_SEASONS}")
    print("  NOTE: historical lineups are the REALIZED 22 -> results are an")
    print("        UPPER BOUND on real pre-game (named-22) signal.\n")

    rows = build_features_fantasy(matches, lineups)

    # ---- 1. raw correlations on the test hold-out ----
    test = _split(rows, TEST_SEASONS, ["elo_diff", "avail_diff", "margin", "home_win"])
    train = _split(rows, TRAIN_SEASONS, ["elo_diff", "avail_diff", "margin", "home_win"])
    n_test = len(test["margin"])

    print(_hr("-"))
    print("[1] CORRELATIONS (test 2024-2025, n=%d)\n" % n_test)
    r_elo = _pearson(test["elo_diff"], test["margin"])
    r_av = _pearson(test["avail_diff"], test["margin"])
    print(f"  corr(elo_diff,    margin) = {r_elo:+.4f}")
    print(f"  corr(avail_diff,  margin) = {r_av:+.4f}  (approach a, fantasy)")

    # ---- 2. partial signal: availability vs margin residual after Elo ----
    elo_only = Ridge(alpha=1.0).fit(train["elo_diff"].reshape(-1, 1), train["margin"])
    resid_test = test["margin"] - elo_only.predict(test["elo_diff"].reshape(-1, 1))
    r_partial = _pearson(test["avail_diff"], resid_test)
    print(f"  corr(avail_diff,  margin residual after Elo) = {r_partial:+.4f}")
    print("    ^ the honest 'beyond team rating' number.\n")

    # ---- 3. incremental ridge (MAE) + logistic (Brier) ----
    print(_hr("-"))
    print("[3] INCREMENTAL VALUE (fit on train, eval on test)\n")

    Xtr_a = train["elo_diff"].reshape(-1, 1)
    Xte_a = test["elo_diff"].reshape(-1, 1)
    Xtr_b = np.column_stack([train["elo_diff"], train["avail_diff"]])
    Xte_b = np.column_stack([test["elo_diff"], test["avail_diff"]])

    ra = Ridge(alpha=1.0).fit(Xtr_a, train["margin"])
    rb = Ridge(alpha=1.0).fit(Xtr_b, train["margin"])
    mae_a = mae(ra.predict(Xte_a), test["margin"])
    mae_b = mae(rb.predict(Xte_b), test["margin"])

    la = LogisticRegression(max_iter=1000).fit(Xtr_a, train["home_win"])
    lb = LogisticRegression(max_iter=1000).fit(Xtr_b, train["home_win"])
    brier_a = brier_score(la.predict_proba(Xte_a)[:, 1], test["home_win"])
    brier_b = brier_score(lb.predict_proba(Xte_b)[:, 1], test["home_win"])

    print(f"  margin MAE   elo-only = {mae_a:7.3f}   elo+avail = {mae_b:7.3f}"
          f"   delta = {mae_b - mae_a:+.3f}")
    print(f"  win Brier    elo-only = {brier_a:7.4f}   elo+avail = {brier_b:7.4f}"
          f"   delta = {brier_b - brier_a:+.4f}")
    print("    (negative delta = availability HELPS)\n")

    # ---- 4. fitted points-per-value scalar ----
    print(_hr("-"))
    print("[4] FITTED fantasy-points -> margin-points scalar\n")
    allr = _split(rows, TRAIN_SEASONS + TEST_SEASONS, ["avail_diff", "margin"])
    uni = Ridge(alpha=0.0, fit_intercept=True).fit(
        allr["avail_diff"].reshape(-1, 1), allr["margin"]
    )
    beta = float(uni.coef_[0])
    beta_ctrl = float(rb.coef_[1])  # avail coef when fit alongside elo_diff
    print(f"  univariate:        margin ~= {beta:+.5f} * avail_diff")
    print(f"  Elo-controlled:    margin += {beta_ctrl:+.5f} * avail_diff")
    print(f"  -> DEFAULT_POINTS_PER_VALUE should be ~{beta_ctrl:.4f} (Elo-controlled).\n")

    # ---- 5. approach (b): ridge player indicators ----
    print(_hr("-"))
    print("[5] APPROACH (b): ridge on per-player presence indicators\n")
    coef = fit_player_ridge(matches, lineups, TRAIN_SEASONS)
    rows_b = build_features_ridge_values(matches, lineups, coef)
    test_b = _split(rows_b, TEST_SEASONS, ["avail_diff", "margin"])
    r_b = _pearson(test_b["avail_diff"], test_b["margin"])
    resid_b = test["margin"]  # same test ordering not guaranteed; recompute cleanly
    # partial for b: residual after elo (reuse elo residual from approach-a test set
    # only valid if same matches/order; both use identical match order + filter)
    if len(test_b["avail_diff"]) == len(resid_test):
        r_b_partial = _pearson(test_b["avail_diff"], resid_test)
    else:
        r_b_partial = float("nan")
    print(f"  trained player coefs: {len(coef)}")
    print(f"  corr(avail_diff_b, margin)               = {r_b:+.4f}")
    print(f"  corr(avail_diff_b, margin resid after Elo)= {r_b_partial:+.4f}")
    print(f"  vs approach (a) partial corr             = {r_partial:+.4f}")
    better = "b (ridge)" if abs(r_b_partial) > abs(r_partial) else "a (fantasy)"
    print(f"  -> stronger availability signal: approach {better}\n")

    # ---- verdict ----
    print(_hr())
    print("[VERDICT]\n")
    helps_mae = mae_b < mae_a
    helps_brier = brier_b < brier_a
    sig = abs(r_partial) > 0.05
    print(f"  availability partial corr (beyond Elo): {r_partial:+.4f}")
    print(f"  helps margin MAE:  {helps_mae}   helps win Brier: {helps_brier}")
    if helps_mae and helps_brier and sig:
        print("  => Lineup value adds REAL signal beyond team rating (on realized")
        print("     lineups). Worth wiring into team_model features.")
    elif helps_mae or helps_brier:
        print("  => Lineup value adds MARGINAL signal. Plausibly useful, but small;")
        print("     real-world (named-22) value will be smaller still.")
    else:
        print("  => No reliable incremental signal in this setup. Be skeptical.")
    print("  REMINDER: realized-lineup optimism inflates all of the above.")
    print(_hr())
    return 0


if __name__ == "__main__":
    sys.exit(main())
