"""Tests for environment and bridge."""

import numpy as np

from src.ingest.team_features import RollingTeamStats
from src.macro.environment import from_team_stats
from src.micro.agent import PlayerAgent
from src.micro.bridge import warp_agent_parameters


def _roll(team: str, i50: float, cp: float, momentum: float) -> RollingTeamStats:
    return RollingTeamStats(
        team=team, games=5, avg_score=85, avg_conceded=80,
        i50_diff=i50, cp_rate=cp, tackle_rate=60, momentum=momentum, win_streak=2,
    )


def test_environment_from_stats():
    env = from_team_stats(_roll("A", 5, 320, 0.3), _roll("B", -3, 300, -0.1))
    assert 0 <= env.pressure_index <= 1
    assert -1 <= env.territory_tilt <= 1


def test_bridge_warps_inside_mid_under_pressure():
    agent = PlayerAgent(
        player_name="Test Mid", team="A", role="Inside Mid",
        disp_floor=10, disp_p50=25, disp_ceil=40,
        goal_lambda=0.5, behind_lambda=0.5, alpha=5, beta=5,
        pressure_sensitivity=0.35, territory_sensitivity=0.15,
    )
    low_pressure = from_team_stats(_roll("A", 0, 250, 0), _roll("B", 0, 250, 0))
    high_pressure = from_team_stats(_roll("A", 0, 400, 0), _roll("B", 0, 400, 0))

    a_low, b_low, _, _ = warp_agent_parameters(agent, low_pressure, is_home=True)
    a_high, b_high, _, _ = warp_agent_parameters(agent, high_pressure, is_home=True)
    assert a_high > a_low
