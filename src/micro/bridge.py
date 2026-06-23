"""Macro to micro bridge — warp player distribution parameters."""

from __future__ import annotations

import numpy as np

from src.macro.environment import EnvironmentState
from src.micro.agent import PlayerAgent


def warp_agent_parameters(
    agent: PlayerAgent,
    env: EnvironmentState,
    is_home: bool,
) -> tuple[float, float, float, float]:
    """
    Warp Beta alpha/beta and Poisson lambdas based on environment.
    Returns (alpha, beta, goal_lambda, behind_lambda).
    """
    momentum = env.home_momentum if is_home else env.away_momentum
    tilt = env.territory_tilt if is_home else -env.territory_tilt

    pressure_shift = env.pressure_index * agent.pressure_sensitivity
    territory_shift = tilt * agent.territory_sensitivity
    momentum_shift = momentum * 0.15

    total_mod = 1.0 + pressure_shift + territory_shift + momentum_shift

    alpha = max(0.1, agent.alpha * total_mod)
    beta = max(0.1, agent.beta * (2.0 - min(total_mod, 1.8)))

    goal_lam = max(0.05, agent.goal_lambda * (1.0 + tilt * 0.3 + momentum * 0.1))
    behind_lam = max(0.05, agent.behind_lambda * (1.0 + tilt * 0.2))

    return alpha, beta, goal_lam, behind_lam


def batch_warp_parameters(
    agents: list[PlayerAgent],
    env: EnvironmentState,
    is_home: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized parameter warping for all agents on one team."""
    alphas, betas, goal_lams, behind_lams = [], [], [], []
    for agent in agents:
        a, b, g, bh = warp_agent_parameters(agent, env, is_home)
        alphas.append(a)
        betas.append(b)
        goal_lams.append(g)
        behind_lams.append(bh)
    return (
        np.array(alphas, dtype=np.float64),
        np.array(betas, dtype=np.float64),
        np.array(goal_lams, dtype=np.float64),
        np.array(behind_lams, dtype=np.float64),
    )
