"""Standalone walk-forward evaluation for the AFL ratings engine.

Compares the original flat-HGA Elo against the upgraded venue/travel + inter-
season regression + recency + offensive/defensive Elo, with an ablation so we
can see which component actually moves the needle.

Honest walk-forward protocol per test season ``S``:
  * Train state is built ONLY from completed matches with ``year < S``.
  * Home-venue shares, the Elo-diff->margin scale and the recency reference
    date all use ``year < S`` data only.
  * Hyper-parameters (HGA base/travel coef, regression factor) are selected
    once on a *training-internal* validation slice (predict the two latest
    pre-test seasons) so nothing from 2024+ ever informs them.
  * Within season ``S`` each match is predicted *before* it is processed, then
    folded in (online update), so no future information leaks.

Run:  python scripts/eval_ratings.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from src.db.models import Match
from src.db.session import SessionLocal
from src.eval.metrics import brier_score, log_loss, mae
from src.macro import venues
from src.macro.elo import (
    RECENCY_K_MULT_DEFAULT,
    RECENCY_WINDOW_DAYS_DEFAULT,
    EloEngine,
    compute_home_venue_shares,
)

TEST_SEASONS = [2024, 2025]


# --------------------------------------------------------------------------- #
# Configuration / engine construction (one fresh engine per configuration)
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    name: str
    venue: bool = False
    regress: bool = False
    recency: bool = False
    dual: bool = False
    base_hga: float = venues.BASE_HGA_DEFAULT
    travel_coef: float = venues.TRAVEL_COEF_DEFAULT
    regress_factor: float = 0.35

    def make(self) -> EloEngine:
        return EloEngine(
            hga_mode="venue" if self.venue else "flat",
            base_hga=self.base_hga,
            travel_coef=self.travel_coef,
            regress_factor=self.regress_factor if self.regress else 0.0,
            recency_k_mult=RECENCY_K_MULT_DEFAULT if self.recency else 1.0,
            recency_window_days=RECENCY_WINDOW_DAYS_DEFAULT,
            dual_ratings=self.dual,
        )


def _completed(matches: list[Match]) -> list[Match]:
    return [m for m in matches if m.home_score is not None and m.away_score is not None]


def _load(session, year_lt: int | None = None, year_eq: int | None = None) -> list[Match]:
    stmt = select(Match).where(Match.complete == True)  # noqa: E712
    if year_lt is not None:
        stmt = stmt.where(Match.year < year_lt)
    if year_eq is not None:
        stmt = stmt.where(Match.year == year_eq)
    stmt = stmt.order_by(Match.year, Match.round, Match.date)
    return _completed(session.scalars(stmt).all())


def _prep(engine: EloEngine, cfg: Config, train: list[Match]) -> None:
    """Attach training-derived state (shares + recency ref) to a fresh engine."""
    if cfg.venue:
        engine.home_venue_shares = compute_home_venue_shares(train)
    if cfg.recency:
        dates = [m.date for m in train if m.date is not None]
        if dates:
            engine.recency_ref_date = max(dates)


def _replay(engine: EloEngine, matches: list[Match]) -> None:
    for m in matches:
        engine.process_match(
            m.home_team, m.away_team, m.home_score, m.away_score,
            venue=m.venue, season=m.year, match_date=m.date,
        )


def fit_margin_scale_for(cfg: Config, train: list[Match]) -> float:
    """Walk-forward fit of predicted_margin ~= scale * elo_diff on train data."""
    engine = cfg.make()
    _prep(engine, cfg, train)
    xs: list[float] = []
    ys: list[float] = []
    for m in train:
        pred = engine.predict(m.home_team, m.away_team, m.venue)
        xs.append(pred["elo_diff"])
        ys.append(m.home_score - m.away_score)
        engine.process_match(
            m.home_team, m.away_team, m.home_score, m.away_score,
            venue=m.venue, season=m.year, match_date=m.date,
        )
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom > 0 else 0.06


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #


@dataclass
class SeasonResult:
    brier: float
    log_loss: float
    mae: float
    dual_mae: float | None
    n: int


def walk_forward(session, cfg: Config, season: int) -> SeasonResult:
    train = _load(session, year_lt=season)
    test = _load(session, year_eq=season)

    scale = fit_margin_scale_for(cfg, train)

    engine = cfg.make()
    _prep(engine, cfg, train)
    _replay(engine, train)

    probs, outcomes, m_pred, m_act, dual_pred = [], [], [], [], []
    for m in test:
        pred = engine.predict(m.home_team, m.away_team, m.venue)
        probs.append(pred["home_win_prob"])
        outcomes.append(1.0 if m.home_score > m.away_score else 0.0)
        m_pred.append(pred["elo_diff"] * scale)
        m_act.append(m.home_score - m.away_score)
        if cfg.dual:
            dual_pred.append(engine.expected_margin(m.home_team, m.away_team, m.venue))
        engine.process_match(
            m.home_team, m.away_team, m.home_score, m.away_score,
            venue=m.venue, season=m.year, match_date=m.date,
        )

    probs_a = np.asarray(probs)
    out_a = np.asarray(outcomes)
    dual_mae = mae(np.asarray(dual_pred), np.asarray(m_act)) if cfg.dual else None
    return SeasonResult(
        brier=brier_score(probs_a, out_a),
        log_loss=log_loss(probs_a, out_a),
        mae=mae(np.asarray(m_pred), np.asarray(m_act)),
        dual_mae=dual_mae,
        n=len(probs),
    )


# --------------------------------------------------------------------------- #
# Hyper-parameter selection (training data only -> no test leakage)
# --------------------------------------------------------------------------- #


def _inner_brier(session, cfg: Config, val_seasons: list[int]) -> float:
    return float(np.mean([walk_forward(session, cfg, s).brier for s in val_seasons]))


def select_hyperparams(session, fit_season: int) -> tuple[float, float, float]:
    """Choose (base_hga, travel_coef, regress_factor) by minimising mean Brier
    on the two latest seasons strictly before ``fit_season``."""
    train_all = _load(session, year_lt=fit_season)
    years = sorted({m.year for m in train_all})
    if len(years) < 3:
        return venues.BASE_HGA_DEFAULT, venues.TRAVEL_COEF_DEFAULT, 0.35
    val_seasons = years[-2:]

    # Stage 1: HGA base/coef (regression on, mid factor) -- but only evaluate on
    # seasons whose own training set excludes them (walk_forward handles that).
    best_hga = (venues.BASE_HGA_DEFAULT, venues.TRAVEL_COEF_DEFAULT)
    best_b = float("inf")
    for base in (25.0, 32.0, 38.0, 44.0, 50.0):
        for coef in (0.0, 0.006, 0.011, 0.016):
            cfg = Config("s1", venue=True, regress=True, base_hga=base,
                         travel_coef=coef, regress_factor=0.35)
            b = _inner_brier(session, cfg, val_seasons)
            if b < best_b:
                best_b, best_hga = b, (base, coef)

    # Stage 2: regression factor at the chosen HGA.
    best_rf, best_b2 = 0.35, float("inf")
    for rf in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
        cfg = Config("s2", venue=True, regress=True, base_hga=best_hga[0],
                     travel_coef=best_hga[1], regress_factor=rf)
        b = _inner_brier(session, cfg, val_seasons)
        if b < best_b2:
            best_b2, best_rf = b, rf

    return best_hga[0], best_hga[1], best_rf


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def main() -> int:
    session = SessionLocal()
    try:
        base_hga, travel_coef, regress_factor = select_hyperparams(
            session, fit_season=TEST_SEASONS[0]
        )
        print("Hyper-parameters selected on training data only "
              f"(validated on the 2 seasons before {TEST_SEASONS[0]}):")
        print(f"  base_hga={base_hga:.1f} Elo pts | "
              f"travel_coef={travel_coef:.3f} Elo pts/km | "
              f"regress_factor={regress_factor:.2f}\n")

        configs = [
            Config("flat Elo (baseline, HGA=50)"),
            Config("+ venue/travel HGA", venue=True,
                   base_hga=base_hga, travel_coef=travel_coef),
            Config("+ venue + regression  [recommended]", venue=True, regress=True,
                   base_hga=base_hga, travel_coef=travel_coef,
                   regress_factor=regress_factor),
            Config("+ recency weighting", venue=True, regress=True,
                   recency=True, base_hga=base_hga, travel_coef=travel_coef,
                   regress_factor=regress_factor),
            Config("FULL (+ off/def, margin only)", venue=True, regress=True,
                   recency=True, dual=True, base_hga=base_hga,
                   travel_coef=travel_coef, regress_factor=regress_factor),
        ]

        results: dict[str, dict[int, SeasonResult]] = {}
        for cfg in configs:
            results[cfg.name] = {s: walk_forward(session, cfg, s) for s in TEST_SEASONS}

        # ---- Headline before/after table (Brier + MAE) ----
        header = f"{'Configuration':<40}"
        for s in TEST_SEASONS:
            header += f" | {s} Brier  {s} LogL  {s} MAE"
        print(header)
        print("-" * len(header))
        for cfg in configs:
            row = f"{cfg.name:<40}"
            for s in TEST_SEASONS:
                r = results[cfg.name][s]
                row += f" |  {r.brier:.4f}  {r.log_loss:.4f}  {r.mae:5.2f}"
            print(row)

        recommended = "+ venue + regression  [recommended]"

        # ---- Dual (attack/defense) margin MAE ----
        print("\nOffensive/Defensive expected-margin MAE (off/def only affects "
              "margin, not win prob):")
        full = results["FULL (+ off/def, margin only)"]
        for s in TEST_SEASONS:
            r = full[s]
            print(f"  {s}: off/def margin MAE = {r.dual_mae:5.2f}  "
                  f"(elo-diff margin MAE = {r.mae:5.2f})")

        # ---- Before/after vs flat baseline ----
        print("\nBefore/after: flat baseline vs recommended "
              "(negative delta = improvement):")
        base = results["flat Elo (baseline, HGA=50)"]
        rec = results[recommended]
        for s in TEST_SEASONS:
            b0, r0 = base[s], rec[s]
            db = r0.brier - b0.brier
            dl = r0.log_loss - b0.log_loss
            dm = r0.mae - b0.mae
            print(f"  {s} (n={b0.n}): "
                  f"Brier {b0.brier:.4f}->{r0.brier:.4f} ({db:+.4f}, {100*db/b0.brier:+.1f}%) | "
                  f"LogL {b0.log_loss:.4f}->{r0.log_loss:.4f} ({dl:+.4f}) | "
                  f"MAE {b0.mae:.2f}->{r0.mae:.2f} ({dm:+.2f})")

        # ---- Marginal Brier contribution of each added component ----
        def avg_brier(name: str) -> float:
            return float(np.mean([results[name][s].brier for s in TEST_SEASONS]))

        print("\nMarginal Brier contribution of each added component "
              "(avg over test seasons):")
        order = [c.name for c in configs]
        prev = avg_brier(order[0])
        print(f"  {'(start) ' + order[0]:<44} {prev:.4f}")
        for name in order[1:]:
            cur = avg_brier(name)
            tag = "  <-- helps" if cur < prev - 1e-5 else ("  <-- hurts" if cur > prev + 1e-5 else "")
            print(f"  {name:<44} {prev:.4f} -> {cur:.4f} ({cur - prev:+.4f}){tag}")
            prev = cur
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
