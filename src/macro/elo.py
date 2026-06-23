"""Modified Elo rating system for AFL.

This module keeps the original scalar Elo (flat HGA) fully intact for backward
compatibility and layers four optional upgrades on top, each independently
toggleable and off by default so existing callers see *identical* behaviour:

1. Venue/travel-aware HGA  (``hga_mode="venue"``, see :mod:`src.macro.venues`).
2. Offensive/defensive (attack/defense) point ratings (``dual_ratings=True``).
3. Inter-season regression to the mean (``regress_factor > 0``).
4. Recency weighting via a higher effective K for recent games
   (``recency_k_mult > 1``).

The public surface used elsewhere is unchanged:
``EloEngine().predict(home, away) -> {home_win_prob, away_win_prob, home_rating,
away_rating, elo_diff}``, ``process_match(home, away, home_score, away_score)``,
``get(team)``, ``snapshot(...)``, ``build_elo_from_history(session, persist=...)``,
plus the module functions ``expected_score``, ``margin_multiplier``,
``update_elo`` and ``fit_margin_scale``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import ELO_HGA, ELO_K, ELO_START
from src.db.models import EloRating, Match
from src.macro import venues

DEFAULT_MARGIN_SCALE = 0.06

# --- Upgrade defaults (only used when the corresponding feature is enabled) ---
REGRESS_FACTOR_DEFAULT = 0.35  # pull toward the mean each new season
RECENCY_K_MULT_DEFAULT = 1.5  # boost K for games inside the recency window
RECENCY_WINDOW_DAYS_DEFAULT = 540  # ~1.5 seasons counts as "recent"
LEAGUE_AVG_SCORE = 89.0  # AFL points-per-team baseline for dual ratings
DUAL_LR = 0.10  # learning rate for attack/defense point ratings
POINTS_HGA_DEFAULT = 9.0  # home edge expressed in points (dual model)


def expected_score(r_a: float, r_b: float) -> float:
    """Expected win probability for team A."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def margin_multiplier(margin: int, elo_diff: float) -> float:
    """FiveThirtyEight-style margin of victory multiplier."""
    return math.log(abs(margin) + 1) * (2.2 / (0.001 * abs(elo_diff) + 2.2))


def update_elo(
    r_home: float,
    r_away: float,
    margin: int,
    k: float = ELO_K,
    hga: float = ELO_HGA,
) -> tuple[float, float]:
    """Update Elo ratings after a match. margin = home - away."""
    r_home_adj = r_home + hga
    elo_diff = r_home_adj - r_away
    exp_home = expected_score(r_home_adj, r_away)

    if margin > 0:
        actual_home = 1.0
    elif margin < 0:
        actual_home = 0.0
    else:
        actual_home = 0.5

    mov = margin_multiplier(margin, elo_diff)
    delta = k * mov * (actual_home - exp_home)

    return r_home + delta, r_away - delta


@dataclass
class EloEngine:
    ratings: dict[str, float] = field(default_factory=dict)
    k: float = ELO_K
    hga: float = ELO_HGA

    # --- Venue/travel HGA -----------------------------------------------------
    hga_mode: str = "flat"  # "flat" | "venue"
    base_hga: float = venues.BASE_HGA_DEFAULT
    travel_coef: float = venues.TRAVEL_COEF_DEFAULT
    # team -> {venue: share of that team's home games at the venue}
    home_venue_shares: dict[str, dict[str, float]] = field(default_factory=dict)

    # --- Inter-season regression ---------------------------------------------
    regress_factor: float = 0.0  # 0 disables; e.g. 0.25 pulls toward the mean
    _last_season: int | None = field(default=None, repr=False)

    # --- Recency weighting ----------------------------------------------------
    recency_k_mult: float = 1.0  # 1.0 disables
    recency_window_days: int = RECENCY_WINDOW_DAYS_DEFAULT
    recency_ref_date: date | None = None  # games within window of this get boost

    # --- Offensive/defensive (dual) ratings ----------------------------------
    dual_ratings: bool = False
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    dual_lr: float = DUAL_LR
    points_hga: float = POINTS_HGA_DEFAULT

    # ------------------------------------------------------------------ scalar
    def get(self, team: str) -> float:
        return self.ratings.get(team, ELO_START)

    # An interstate "home" venue counts as a genuine home ground once a team
    # stages at least this share of its home games there (e.g. Hawthorn in
    # Tasmania, GWS in Canberra). Below it, the crowd edge is scaled down.
    _SECONDARY_HOME_SHARE = 0.30

    def _home_familiarity(self, home: str, venue: str | None) -> float:
        """Share of the crowd/familiarity HGA that applies for ``home`` here.

        * Any venue in the home team's own state -> full familiarity (a club
          that splits home games across, say, the MCG and Marvel is fully at
          home at both).
        * Interstate venues -> scaled by how established the venue is as a home
          ground for that team, from historical home-game frequency.
        """
        if not venue:
            return 1.0
        if not venues.is_interstate(home, venue):
            return 1.0
        shares = self.home_venue_shares.get(home)
        if not shares:
            return 0.0
        return min(1.0, shares.get(venue, 0.0) / self._SECONDARY_HOME_SHARE)

    def hga_for(self, home: str, away: str, venue: str | None = None) -> float:
        """Resolve the HGA (Elo points) to apply for this matchup/venue."""
        if self.hga_mode == "venue":
            return venues.effective_hga(
                home,
                away,
                venue,
                base_hga=self.base_hga,
                travel_coef=self.travel_coef,
                home_familiarity=self._home_familiarity(home, venue),
            )
        return self.hga

    def predict(self, home: str, away: str, venue: str | None = None) -> dict[str, float]:
        r_home = self.get(home)
        r_away = self.get(away)
        hga = self.hga_for(home, away, venue)
        r_home_adj = r_home + hga
        p_home = expected_score(r_home_adj, r_away)
        return {
            "home_win_prob": p_home,
            "away_win_prob": 1.0 - p_home,
            "home_rating": r_home,
            "away_rating": r_away,
            "elo_diff": r_home_adj - r_away,
        }

    # ------------------------------------------------------------------- dual
    def get_attack(self, team: str) -> float:
        return self.attack.get(team, 0.0)

    def get_defense(self, team: str) -> float:
        return self.defense.get(team, 0.0)

    def expected_scores(
        self, home: str, away: str, venue: str | None = None
    ) -> tuple[float, float]:
        """Expected (home_points, away_points) from attack/defense ratings.

        ``attack``/``defense`` are stored as deviations (in points) from the
        league average. A positive defense rating means the team concedes fewer
        points than average. The home points edge scales with how much of the
        flat points HGA the venue/travel model thinks applies.
        """
        hga_scale = 1.0
        if self.hga_mode == "venue":
            hga_scale = self.hga_for(home, away, venue) / max(self.base_hga, 1e-6)
        edge = self.points_hga * hga_scale / 2.0
        home_pts = LEAGUE_AVG_SCORE + self.get_attack(home) - self.get_defense(away) + edge
        away_pts = LEAGUE_AVG_SCORE + self.get_attack(away) - self.get_defense(home) - edge
        return home_pts, away_pts

    def expected_margin(self, home: str, away: str, venue: str | None = None) -> float:
        home_pts, away_pts = self.expected_scores(home, away, venue)
        return home_pts - away_pts

    def _update_dual(
        self, home: str, away: str, home_score: int, away_score: int, venue: str | None
    ) -> None:
        exp_home, exp_away = self.expected_scores(home, away, venue)
        err_home = home_score - exp_home
        err_away = away_score - exp_away
        lr = self.dual_lr
        # Home score is driven by home attack and away defense.
        self.attack[home] = self.get_attack(home) + lr * err_home
        self.defense[away] = self.get_defense(away) - lr * err_home
        # Away score is driven by away attack and home defense.
        self.attack[away] = self.get_attack(away) + lr * err_away
        self.defense[home] = self.get_defense(home) - lr * err_away

    # --------------------------------------------------------------- seasons
    def regress_to_mean(self) -> None:
        """Pull all ratings ``regress_factor`` of the way back to the mean."""
        if self.regress_factor <= 0:
            return
        keep = 1.0 - self.regress_factor
        for team, r in self.ratings.items():
            self.ratings[team] = ELO_START + (r - ELO_START) * keep
        for team, a in self.attack.items():
            self.attack[team] = a * keep
        for team, d in self.defense.items():
            self.defense[team] = d * keep

    def _k_for(self, match_date: date | None) -> float:
        if (
            self.recency_k_mult != 1.0
            and self.recency_ref_date is not None
            and match_date is not None
        ):
            age_days = (self.recency_ref_date - match_date).days
            if 0 <= age_days <= self.recency_window_days:
                return self.k * self.recency_k_mult
        return self.k

    def process_match(
        self,
        home: str,
        away: str,
        home_score: int,
        away_score: int,
        venue: str | None = None,
        season: int | None = None,
        match_date: date | None = None,
    ) -> None:
        # Inter-season regression fires on the first match of a new season.
        if season is not None and self.regress_factor > 0:
            if self._last_season is not None and season != self._last_season:
                self.regress_to_mean()
            self._last_season = season

        margin = home_score - away_score
        r_home = self.get(home)
        r_away = self.get(away)
        hga = self.hga_for(home, away, venue)
        k = self._k_for(match_date)
        new_home, new_away = update_elo(r_home, r_away, margin, k, hga)
        self.ratings[home] = new_home
        self.ratings[away] = new_away

        if self.dual_ratings:
            self._update_dual(home, away, home_score, away_score, venue)

    def snapshot(
        self,
        as_of: date,
        session: Session,
        teams: list[str] | None = None,
    ) -> None:
        target = teams if teams is not None else list(self.ratings.keys())
        for team in target:
            rating = self.ratings.get(team)
            if rating is None:
                continue
            existing = session.scalars(
                select(EloRating).where(
                    EloRating.team == team, EloRating.as_of_date == as_of
                )
            ).first()
            if existing:
                existing.rating = rating
            else:
                session.add(EloRating(team=team, rating=rating, as_of_date=as_of))

    # ----------------------------------------------------------- convenience
    @classmethod
    def upgraded(
        cls,
        *,
        base_hga: float = venues.BASE_HGA_DEFAULT,
        travel_coef: float = venues.TRAVEL_COEF_DEFAULT,
        regress_factor: float = REGRESS_FACTOR_DEFAULT,
        recency_k_mult: float = 1.0,
        recency_window_days: int = RECENCY_WINDOW_DAYS_DEFAULT,
        dual_ratings: bool = True,
        k: float = ELO_K,
    ) -> "EloEngine":
        """Factory for the recommended upgraded engine (venue/travel HGA +
        inter-season regression + dual attack/defense ratings).

        Recency K-boosting is available but defaults OFF (``recency_k_mult=1.0``)
        because it degraded walk-forward Brier in backtests; pass a value > 1 to
        enable it. Any component can be neutralised via its identity value
        (``travel_coef=0``/``regress_factor=0``/``recency_k_mult=1.0``)."""
        return cls(
            k=k,
            hga_mode="venue",
            base_hga=base_hga,
            travel_coef=travel_coef,
            regress_factor=regress_factor,
            recency_k_mult=recency_k_mult,
            recency_window_days=recency_window_days,
            dual_ratings=dual_ratings,
        )


def compute_home_venue_shares(
    matches: list[Match],
) -> dict[str, dict[str, float]]:
    """From completed matches, the share of each team's home games per venue.

    Used to derive *which venues a team actually plays "home" at* so the HGA
    crowd component can be down-weighted at neutral grounds. Pass only matches
    strictly prior to the prediction window to avoid leakage.
    """
    counts: dict[str, dict[str, int]] = {}
    for m in matches:
        if not m.venue:
            continue
        team_counts = counts.setdefault(m.home_team, {})
        team_counts[m.venue] = team_counts.get(m.venue, 0) + 1
    shares: dict[str, dict[str, float]] = {}
    for team, vc in counts.items():
        total = sum(vc.values())
        if total <= 0:
            continue
        shares[team] = {v: c / total for v, c in vc.items()}
    return shares


def fit_margin_scale(
    session: Session,
    seasons: list[int],
    default: float = DEFAULT_MARGIN_SCALE,
) -> float:
    """Fit the Elo-diff -> margin scalar by regressing actual margins on
    walk-forward Elo diff over the given seasons (through the origin).

    Returns ``k`` such that ``predicted_margin ~= k * elo_diff``. Falls back to
    ``default`` if there is insufficient data.
    """
    if not seasons:
        return default

    elo = EloEngine()
    matches = session.scalars(
        select(Match)
        .where(Match.complete == True, Match.year.in_(seasons))  # noqa: E712
        .order_by(Match.year, Match.round)
    ).all()

    xs: list[float] = []
    ys: list[float] = []
    for m in matches:
        if m.home_score is None or m.away_score is None:
            continue
        pred = elo.predict(m.home_team, m.away_team)
        xs.append(pred["elo_diff"])
        ys.append(m.home_score - m.away_score)
        elo.process_match(m.home_team, m.away_team, m.home_score, m.away_score)

    if len(xs) < 50:
        return default

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    denom = float(np.sum(x * x))
    if denom <= 0:
        return default
    return float(np.sum(x * y) / denom)


def build_elo_from_history(
    session: Session,
    persist: bool = True,
    engine: EloEngine | None = None,
) -> EloEngine:
    """Replay all completed matches chronologically to build current Elo.

    Backward compatible: with no ``engine`` a default (flat) :class:`EloEngine`
    is used, reproducing the original behaviour exactly. Pass a pre-configured
    engine (e.g. ``EloEngine.upgraded()``) to build venue/travel + regression +
    recency + dual ratings instead. Venue, season and match date are forwarded
    to ``process_match`` so the engine's enabled upgrades take effect.
    """
    if engine is None:
        engine = EloEngine()

    matches = session.scalars(
        select(Match)
        .where(Match.complete == True)  # noqa: E712
        .order_by(Match.year, Match.round, Match.date)
    ).all()

    # Recency weighting needs a reference "now": the most recent match date.
    if engine.recency_k_mult != 1.0 and engine.recency_ref_date is None:
        dates = [m.date for m in matches if m.date is not None]
        if dates:
            engine.recency_ref_date = max(dates)

    # Derive home-venue shares from history when running venue-aware HGA.
    if engine.hga_mode == "venue" and not engine.home_venue_shares:
        engine.home_venue_shares = compute_home_venue_shares(list(matches))

    for m in matches:
        if m.home_score is None or m.away_score is None:
            continue
        engine.process_match(
            m.home_team,
            m.away_team,
            m.home_score,
            m.away_score,
            venue=m.venue,
            season=m.year,
            match_date=m.date,
        )
        if persist and m.date:
            engine.snapshot(m.date, session, teams=[m.home_team, m.away_team])

    return engine
