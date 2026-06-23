"""Environment state derived from real team features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ingest.team_features import RollingTeamStats


@dataclass
class EnvironmentState:
    match_id: int | None
    pressure_index: float
    territory_tilt: float
    home_momentum: float
    away_momentum: float
    home_sentiment: float = 0.0
    away_sentiment: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pressure_index": self.pressure_index,
            "territory_tilt": self.territory_tilt,
            "home_momentum": self.home_momentum,
            "away_momentum": self.away_momentum,
            "home_sentiment": self.home_sentiment,
            "away_sentiment": self.away_sentiment,
        }


def from_team_stats(
    home_roll: RollingTeamStats,
    away_roll: RollingTeamStats,
    match_id: int | None = None,
    home_sentiment: float = 0.0,
    away_sentiment: float = 0.0,
) -> EnvironmentState:
    """Compute EnvironmentState from rolling team statistics."""
    cp_rate = (home_roll.cp_rate + away_roll.cp_rate) / 2.0
    tackle_rate = (home_roll.tackle_rate + away_roll.tackle_rate) / 2.0
    combined = cp_rate + tackle_rate * 0.5

    pressure = float(np.clip((combined - 280) / 120, 0, 1))
    tilt = float(np.clip((home_roll.i50_diff - away_roll.i50_diff) / 20, -1, 1))

    home_momentum = float(np.clip(home_roll.momentum + home_sentiment * 0.1, -1, 1))
    away_momentum = float(np.clip(away_roll.momentum + away_sentiment * 0.1, -1, 1))

    return EnvironmentState(
        match_id=match_id,
        pressure_index=pressure,
        territory_tilt=tilt,
        home_momentum=home_momentum,
        away_momentum=away_momentum,
        home_sentiment=home_sentiment,
        away_sentiment=away_sentiment,
    )
