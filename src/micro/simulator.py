"""Numba-optimized Monte Carlo simulation engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.stats import beta as beta_dist

from src.config import DEFAULT_SIMS, PLAYER_BLEND_WEIGHT
from src.macro.environment import EnvironmentState
from src.micro.agent import PlayerAgent
from src.micro.bridge import batch_warp_parameters
from src.micro.copula import _build_correlation_matrix, role_index_map, sample_correlated_uniforms


def _scale_scoring_rates(
    goal_lams: np.ndarray,
    behind_lams: np.ndarray,
    target_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale Poisson rates so expected scoring output matches team target."""
    expected = float(np.sum(goal_lams * 6.0 + behind_lams))
    if expected <= 0 or target_score <= 0:
        return goal_lams, behind_lams
    scale = target_score / expected
    return goal_lams * scale, behind_lams * scale


def _simulate_team_scores(
    goal_lams: np.ndarray,
    behind_lams: np.ndarray,
    n_sims: int,
) -> np.ndarray:
    """Simulate team total points from per-player Poisson rates.

    Vectorized with NumPy's global RNG so results are reproducible when the
    RNG is seeded (Numba's ``parallel=True`` RNG could not be seeded reliably).
    """
    goal_lams = np.ascontiguousarray(goal_lams, dtype=np.float64)
    behind_lams = np.ascontiguousarray(behind_lams, dtype=np.float64)
    goals = np.random.poisson(goal_lams[None, :], size=(n_sims, goal_lams.shape[0]))
    behinds = np.random.poisson(behind_lams[None, :], size=(n_sims, behind_lams.shape[0]))
    return (goals * 6 + behinds).sum(axis=1).astype(np.int32)


@njit(cache=True)
def _blend_scores(player_scores: np.ndarray, team_scores: np.ndarray, weight: float) -> np.ndarray:
    blended = np.empty_like(player_scores, dtype=np.float64)
    for i in range(len(player_scores)):
        blended[i] = weight * player_scores[i] + (1.0 - weight) * team_scores[i]
    return blended.astype(np.int32)


@dataclass
class SimulationResult:
    home_win_prob: float
    away_win_prob: float
    median_home_score: float
    median_away_score: float
    median_margin: float
    std_margin: float
    p95_margin: float
    home_scores: np.ndarray
    away_scores: np.ndarray
    margins: np.ndarray
    player_home_disposals: dict[str, dict[str, float]] | None = None
    player_away_disposals: dict[str, dict[str, float]] | None = None


class HybridSimulator:
    """SFN-Monte Carlo hybrid simulator."""

    def __init__(
        self,
        env: EnvironmentState,
        home_agents: list[PlayerAgent],
        away_agents: list[PlayerAgent],
        team_home_score: float,
        team_away_score: float,
        player_blend: float = PLAYER_BLEND_WEIGHT,
    ):
        self.env = env
        self.home_agents = home_agents
        self.away_agents = away_agents
        self.team_home_score = team_home_score
        self.team_away_score = team_away_score
        self.player_blend = player_blend

    def _sample_disposals(
        self,
        agents: list[PlayerAgent],
        env: EnvironmentState,
        is_home: bool,
        n_sims: int,
    ) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        if not agents:
            return np.zeros(n_sims, dtype=np.int32), {}

        alphas, betas, goal_lams, behind_lams = batch_warp_parameters(agents, env, is_home)
        roles = [a.role for a in agents]
        corr = _build_correlation_matrix(len(agents), role_index_map(roles))
        uniforms = sample_correlated_uniforms(len(agents), n_sims, corr)

        # Clip to avoid inf/nan from ppf at the open interval boundaries.
        uniforms = np.clip(uniforms, 1e-6, 1.0 - 1e-6)

        disp_matrix = np.zeros((n_sims, len(agents)), dtype=np.int32)
        for j, agent in enumerate(agents):
            range_size = max(1, agent.disp_ceil - agent.disp_floor)
            # Map the correlated uniforms through the Beta inverse-CDF so the
            # Gaussian copula's correlation structure is preserved.
            beta_samples = beta_dist.ppf(uniforms[:, j], alphas[j], betas[j])
            disp_matrix[:, j] = (agent.disp_floor + beta_samples * range_size).astype(np.int32)

        player_stats = {}
        for j, agent in enumerate(agents):
            col = disp_matrix[:, j]
            player_stats[agent.player_name] = {
                "p10": float(np.percentile(col, 10)),
                "p50": float(np.percentile(col, 50)),
                "p90": float(np.percentile(col, 90)),
            }

        return disp_matrix, player_stats

    def run(self, n_sims: int = DEFAULT_SIMS, seed: int | None = None) -> SimulationResult:
        if seed is not None:
            # Seed NumPy's global RNG. All stochastic draws (Poisson scores,
            # the np.random.normal team leg, and the copula's multivariate
            # normal) now flow through this single seedable state.
            np.random.seed(seed)

        if self.home_agents:
            _, _, home_goal_lams, home_behind_lams = batch_warp_parameters(
                self.home_agents, self.env, is_home=True
            )
        else:
            home_goal_lams = np.array([self.team_home_score / 30.0])
            home_behind_lams = np.array([self.team_home_score / 40.0])

        if self.away_agents:
            _, _, away_goal_lams, away_behind_lams = batch_warp_parameters(
                self.away_agents, self.env, is_home=False
            )
        else:
            away_goal_lams = np.array([self.team_away_score / 30.0])
            away_behind_lams = np.array([self.team_away_score / 40.0])

        home_goal_lams, home_behind_lams = _scale_scoring_rates(
            home_goal_lams, home_behind_lams, self.team_home_score
        )
        away_goal_lams, away_behind_lams = _scale_scoring_rates(
            away_goal_lams, away_behind_lams, self.team_away_score
        )

        home_player_scores = _simulate_team_scores(home_goal_lams, home_behind_lams, n_sims)
        away_player_scores = _simulate_team_scores(away_goal_lams, away_behind_lams, n_sims)

        # Team leg: normal around ridge prediction
        home_team_leg = np.random.normal(self.team_home_score, 18, n_sims).astype(np.int32)
        away_team_leg = np.random.normal(self.team_away_score, 18, n_sims).astype(np.int32)
        home_team_leg = np.clip(home_team_leg, 20, 200)
        away_team_leg = np.clip(away_team_leg, 20, 200)

        home_final = _blend_scores(home_player_scores, home_team_leg, self.player_blend)
        away_final = _blend_scores(away_player_scores, away_team_leg, self.player_blend)

        margins = home_final.astype(np.float64) - away_final.astype(np.float64)
        home_wins = np.sum(home_final > away_final)

        _, home_disp_stats = self._sample_disposals(
            self.home_agents, self.env, True, min(n_sims, 5000)
        )
        _, away_disp_stats = self._sample_disposals(
            self.away_agents, self.env, False, min(n_sims, 5000)
        )

        return SimulationResult(
            home_win_prob=float(home_wins / n_sims),
            away_win_prob=float((n_sims - home_wins) / n_sims),
            median_home_score=float(np.median(home_final)),
            median_away_score=float(np.median(away_final)),
            median_margin=float(np.median(margins)),
            std_margin=float(np.std(margins)),
            p95_margin=float(np.percentile(margins, 95)),
            home_scores=home_final,
            away_scores=away_final,
            margins=margins,
            player_home_disposals=home_disp_stats,
            player_away_disposals=away_disp_stats,
        )


def hybrid_backtest_margin(
    n_sims: int,
    home_agents: list[PlayerAgent],
    away_agents: list[PlayerAgent],
    env: EnvironmentState,
    team_home: float,
    team_away: float,
) -> tuple[float, float]:
    """Quick hybrid simulation returning (home_win_prob, expected_margin)."""
    sim = HybridSimulator(env, home_agents, away_agents, team_home, team_away)
    result = sim.run(n_sims=n_sims)
    return result.home_win_prob, result.median_margin
