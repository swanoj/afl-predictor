"""Ridge regression team margin model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.ingest.team_features import compute_rolling_stats
from src.macro.elo import DEFAULT_MARGIN_SCALE, EloEngine, build_elo_from_history, fit_margin_scale

# Reasonable default for AFL margin residual spread when a model is unfitted.
DEFAULT_MARGIN_SIGMA = 37.0

# Canonical feature columns shared by every macro win-probability model
# (Ridge margin model, direct logistic, and gradient boosting). Keeping a
# single ordered list guarantees the models are compared on identical inputs.
FEATURE_COLUMNS: list[str] = [
    "elo_diff",
    "i50_diff",
    "cp_diff",
    "home_advantage",
    "form_diff",
    "score_diff",
]

# Enriched feature set used by the stacked "integrated" model. It is the
# canonical set PLUS two cheap signals (player availability and a rolling
# xScore/scoring-shot differential), with ``elo_diff`` sourced from the
# UPGRADED rating engine (venue/travel HGA + inter-season regression) rather
# than the flat one. Kept as a separate list (rather than mutating
# ``FEATURE_COLUMNS``) so the existing logistic/ridge/GBM baselines stay
# untouched and remain a fair comparison point for the integrated model.
INTEGRATED_FEATURE_COLUMNS: list[str] = [
    "elo_diff",
    "i50_diff",
    "cp_diff",
    "home_advantage",
    "form_diff",
    "score_diff",
    "avail_diff",
    "xscore_diff",
]

# Recommended upgraded-engine hyper-parameters for the integrated rating source
# (the "big win" upgrade): stronger inter-season regression to the mean plus a
# larger, venue/travel-aware home-ground advantage.
UPGRADED_ELO_KWARGS: dict[str, float] = {
    "regress_factor": 0.60,
    "base_hga": 50.0,
    "travel_coef": 0.016,
}

# Rolling window (in matches) for the xScore differential feature. Window 8
# was the sweet spot in scripts/xscore_research_notes.md.
XSCORE_WINDOW = 8

# League average points per scoring shot. xScore is a pure rescale of scoring
# shots, so the exact constant is immaterial once the logistic standardises its
# inputs; we keep it so the feature is expressed in interpretable "points".
LEAGUE_PTS_PER_SHOT = 3.99


def compute_match_features(
    session: Session,
    match,
    elo_engine: EloEngine,
) -> dict[str, float]:
    """Build the macro feature vector for a single match.

    Uses the Elo engine's *current* state (the caller is responsible for
    walking it forward), plus rolling team stats computed strictly from
    matches before this one. Shared by ``TeamMarginModel`` and the direct
    win-probability models so they all train on identical features.
    """
    home_roll = compute_rolling_stats(session, match.home_team, match.year, match.round)
    away_roll = compute_rolling_stats(session, match.away_team, match.year, match.round)
    pred = elo_engine.predict(match.home_team, match.away_team)

    return {
        "elo_diff": pred["elo_diff"],
        "i50_diff": home_roll.i50_diff - away_roll.i50_diff,
        "cp_diff": home_roll.cp_rate - away_roll.cp_rate,
        "home_advantage": 1.0,
        "form_diff": home_roll.momentum - away_roll.momentum,
        "score_diff": home_roll.avg_score - away_roll.avg_conceded,
    }


# Process-level cache for the full walk-forward feature frame. The frame is a
# pure function of the (static) match history in the DB, so we build it once
# per run and slice it for every model/season. Call ``clear_feature_cache()``
# in tests that mutate match data.
_FRAME_CACHE: dict[int, pd.DataFrame] = {}
_INTEGRATED_FRAME_CACHE: dict[int, pd.DataFrame] = {}


def clear_feature_cache() -> None:
    _FRAME_CACHE.clear()
    _INTEGRATED_FRAME_CACHE.clear()


def build_full_feature_frame(session: Session, use_cache: bool = True) -> pd.DataFrame:
    """Walk-forward feature frame over *all* completed matches.

    Returns one row per completed match with the canonical features plus
    metadata (``squiggle_id``, ``year``, ``round``, teams) and both targets
    (``margin`` and binary ``home_win``). Elo is updated strictly after each
    match is recorded, so every row is leakage-free and identical to what an
    online predictor would have seen. Models obtain their train/test/
    calibration splits by slicing this frame on ``year``.
    """
    from src.db.models import Match

    cache_key = id(session.get_bind())
    if use_cache and cache_key in _FRAME_CACHE:
        return _FRAME_CACHE[cache_key]

    elo = EloEngine()
    matches = session.scalars(
        select(Match)
        .where(Match.complete == True)  # noqa: E712
        .order_by(Match.year, Match.round)
    ).all()

    rows: list[dict] = []
    for m in matches:
        if m.home_score is None or m.away_score is None:
            continue
        feats = compute_match_features(session, m, elo)
        margin = m.home_score - m.away_score
        rows.append(
            {
                **feats,
                "squiggle_id": m.squiggle_id,
                "year": m.year,
                "round": m.round,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "margin": margin,
                "home_win": 1.0 if margin > 0 else 0.0,
            }
        )
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    frame = pd.DataFrame(rows)
    if use_cache:
        _FRAME_CACHE[cache_key] = frame
    return frame


def build_integrated_feature_frame(session: Session, use_cache: bool = True) -> pd.DataFrame:
    """Walk-forward feature frame for the stacked *integrated* model.

    One row per completed match. Identical metadata/targets to
    :func:`build_full_feature_frame`, plus the enriched feature set:

    * ``elo_diff``      — from the UPGRADED engine (venue/travel HGA +
      inter-season regression), venue-aware.
    * ``elo_diff_flat`` — the flat-HGA Elo diff, kept ONLY so the ablation can
      swap the upgraded rating source out for the flat one.
    * ``i50_diff`` / ``cp_diff`` / ``home_advantage`` / ``form_diff`` /
      ``score_diff`` — the canonical rolling features (unchanged).
    * ``avail_diff``    — home minus away player-availability adjustment (margin
      points) from the realized lineup. See the OPTIMISM CAVEAT in
      :mod:`src.ingest.lineups`: on historical data this is an UPPER BOUND on
      the live signal because it uses who *actually* played, not the named 22.
    * ``xscore_diff``   — home minus away rolling (last ``XSCORE_WINDOW`` games)
      xScore margin, where xScore = scoring_shots * ``LEAGUE_PTS_PER_SHOT`` and
      scoring_shots = goals + behinds summed from ``PlayerGameLog``.

    Every feature for a given match uses ONLY data strictly before that match
    (ratings walked forward online; rolling stats/lineups/scoring-shots filtered
    to prior games; player values use the leakage-free per-season snapshot), so
    slicing the frame by ``year`` is leak-free for predicting any season.
    """
    from src.db.models import Match, PlayerGameLog
    from src.ingest.lineups import DEFAULT_BASELINE_WINDOW, lineup_value
    from src.micro.player_value import (
        DEFAULT_POINTS_PER_VALUE,
        load_player_values,
        replacement_value,
    )

    cache_key = id(session.get_bind())
    if use_cache and cache_key in _INTEGRATED_FRAME_CACHE:
        return _INTEGRATED_FRAME_CACHE[cache_key]

    matches = session.scalars(
        select(Match)
        .where(Match.complete == True)  # noqa: E712
        .order_by(Match.year, Match.round, Match.date)
    ).all()

    # --- Bulk-load player logs once: realized rosters + scoring shots. -------
    roster: dict[tuple[int, str], list[str]] = {}
    shots: dict[tuple[int, str], float] = {}
    for log in session.scalars(select(PlayerGameLog)).all():
        key = (log.match_id, log.team)
        roster.setdefault(key, []).append(log.player_name)
        shots[key] = shots.get(key, 0.0) + (log.goals or 0) + (log.behinds or 0)

    # --- Walk-forward state. -------------------------------------------------
    upgraded = EloEngine.upgraded(**UPGRADED_ELO_KWARGS)
    upgraded.home_venue_shares = {}  # built incrementally below (leak-free)
    flat = EloEngine()

    # Running per-team home-venue counts -> shares, updated AFTER each match so a
    # match's HGA familiarity only ever reflects strictly-prior home games.
    venue_counts: dict[str, dict[str, int]] = {}
    # Per-team chronological history for rolling features (strictly prior games).
    shot_hist: dict[str, list[float]] = {}
    prior_match_ids: dict[str, list[int]] = {}
    # Per-season leakage-free player-value snapshots (computed lazily, cached).
    values_by_season: dict[int, dict] = {}

    def _values(year: int) -> dict:
        if year not in values_by_season:
            values_by_season[year] = load_player_values(session, year)
        return values_by_season[year]

    def _avail(team: str, year: int) -> float:
        """Realized-lineup availability adjustment (margin points) vs baseline.

        Mirrors ``lineups.availability_adjustment`` with ``expected_lineup=None``
        but reuses the bulk-loaded rosters/value snapshots for speed.
        """
        vals = _values(year)
        replacement = replacement_value(vals)
        current_names = roster.get((mid, team))
        if not current_names:
            return 0.0
        current = lineup_value(current_names, team, vals, replacement)
        recent = prior_match_ids.get(team, [])[-DEFAULT_BASELINE_WINDOW:]
        baseline_vals = [
            lineup_value(roster[(pid, team)], team, vals, replacement)
            for pid in recent
            if roster.get((pid, team))
        ]
        if not baseline_vals:
            return 0.0
        baseline = sum(baseline_vals) / len(baseline_vals)
        return (current - baseline) * DEFAULT_POINTS_PER_VALUE

    def _roll_xscore(team: str) -> float:
        hist = shot_hist.get(team)
        if not hist:
            return 0.0
        recent = hist[-XSCORE_WINDOW:]
        return sum(recent) / len(recent)

    rows: list[dict] = []
    for m in matches:
        if m.home_score is None or m.away_score is None:
            continue
        mid = m.id
        home, away = m.home_team, m.away_team

        home_roll = compute_rolling_stats(session, home, m.year, m.round)
        away_roll = compute_rolling_stats(session, away, m.year, m.round)

        up = upgraded.predict(home, away, venue=m.venue)
        fl = flat.predict(home, away)

        avail_diff = _avail(home, m.year) - _avail(away, m.year)
        xscore_diff = _roll_xscore(home) - _roll_xscore(away)

        margin = m.home_score - m.away_score
        rows.append(
            {
                "elo_diff": up["elo_diff"],
                "elo_diff_flat": fl["elo_diff"],
                "i50_diff": home_roll.i50_diff - away_roll.i50_diff,
                "cp_diff": home_roll.cp_rate - away_roll.cp_rate,
                "home_advantage": 1.0,
                "form_diff": home_roll.momentum - away_roll.momentum,
                "score_diff": home_roll.avg_score - away_roll.avg_conceded,
                "avail_diff": avail_diff,
                "xscore_diff": xscore_diff,
                "squiggle_id": m.squiggle_id,
                "year": m.year,
                "round": m.round,
                "home_team": home,
                "away_team": away,
                "margin": margin,
                "home_win": 1.0 if margin > 0 else 0.0,
            }
        )

        # --- Advance walk-forward state AFTER recording the row (no leakage). -
        upgraded.process_match(
            home, away, m.home_score, m.away_score,
            venue=m.venue, season=m.year, match_date=m.date,
        )
        flat.process_match(home, away, m.home_score, m.away_score)

        if m.venue:
            vc = venue_counts.setdefault(home, {})
            vc[m.venue] = vc.get(m.venue, 0) + 1
            total = sum(vc.values())
            upgraded.home_venue_shares[home] = {v: c / total for v, c in vc.items()}

        sf = shots.get((mid, home))
        sa = shots.get((mid, away))
        if sf is not None and sa is not None:
            shot_hist.setdefault(home, []).append((sf - sa) * LEAGUE_PTS_PER_SHOT)
            shot_hist.setdefault(away, []).append((sa - sf) * LEAGUE_PTS_PER_SHOT)
        prior_match_ids.setdefault(home, []).append(mid)
        prior_match_ids.setdefault(away, []).append(mid)

    frame = pd.DataFrame(rows)
    if use_cache:
        _INTEGRATED_FRAME_CACHE[cache_key] = frame
    return frame


@dataclass
class TeamMarginModel:
    model: Ridge = field(default_factory=lambda: Ridge(alpha=1.0))
    fitted: bool = False
    margin_scale: float = DEFAULT_MARGIN_SCALE
    residual_std: float = DEFAULT_MARGIN_SIGMA

    def _build_features(
        self,
        session: Session,
        match,
        elo_engine: EloEngine,
    ) -> dict[str, float]:
        return compute_match_features(session, match, elo_engine)

    def build_training_frame(self, session: Session, seasons: list[int]) -> tuple[pd.DataFrame, pd.Series]:
        from src.db.models import Match

        elo = EloEngine()
        rows = []
        targets = []

        matches = session.scalars(
            select(Match)
            .where(Match.complete == True, Match.year.in_(seasons))  # noqa: E712
            .order_by(Match.year, Match.round)
        ).all()

        for m in matches:
            if m.home_score is None or m.away_score is None:
                continue
            feats = self._build_features(session, m, elo)
            rows.append(feats)
            targets.append(m.home_score - m.away_score)
            elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

        X = pd.DataFrame(rows)
        y = pd.Series(targets, name="margin")
        return X, y

    def fit(self, session: Session, seasons: list[int]) -> None:
        X, y = self.build_training_frame(session, seasons)
        if len(X) < 50:
            raise ValueError("Insufficient training data for Ridge model")
        self.model.fit(X, y)
        self.fitted = True

        # Fit the Elo-diff -> margin scalar (used as fallback) on the same data.
        self.margin_scale = fit_margin_scale(session, seasons)

        # Residual spread of predicted margins -> used to turn the model's
        # margin into an independent win probability via the normal CDF.
        resid = y.to_numpy() - self.model.predict(X)
        std = float(np.std(resid))
        self.residual_std = std if std > 1e-6 else DEFAULT_MARGIN_SIGMA

    def predict_margin(self, session: Session, match, elo_engine: EloEngine) -> float:
        feats = self._build_features(session, match, elo_engine)
        X = pd.DataFrame([feats])
        if not self.fitted:
            return feats["elo_diff"] * self.margin_scale  # rough fallback
        return float(self.model.predict(X)[0])

    def predict_win_prob(self, margin: float) -> float:
        """Independent home-win probability derived from the model's own
        predicted margin and the training-set residual spread."""
        sigma = self.residual_std if self.residual_std > 1e-6 else DEFAULT_MARGIN_SIGMA
        return float(norm.cdf(margin / sigma))

    def predict_scores(self, session: Session, match, elo_engine: EloEngine) -> tuple[float, float]:
        """Predict home and away scores from margin and league average."""
        margin = self.predict_margin(session, match, elo_engine)
        home_roll = compute_rolling_stats(session, match.home_team, match.year, match.round)
        away_roll = compute_rolling_stats(session, match.away_team, match.year, match.round)
        league_avg = (home_roll.avg_score + away_roll.avg_score) / 2
        home_score = league_avg + margin / 2
        away_score = league_avg - margin / 2
        return max(home_score, 30), max(away_score, 30)
