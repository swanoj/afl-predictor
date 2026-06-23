"""Gaussian copula for correlated player performance."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _build_correlation_matrix(n: int, role_indices: dict[str, list[int]]) -> np.ndarray:
    """Build correlation matrix based on role groupings."""
    corr = np.eye(n)
    ruck_mids = 0.35
    mid_fwd = 0.25
    def_fwd = -0.15

    rucks = role_indices.get("Ruck", [])
    mids = role_indices.get("Inside Mid", []) + role_indices.get("Outside Mid", [])
    forwards = role_indices.get("Key Forward", []) + role_indices.get("Forward", [])
    defenders = role_indices.get("Key Defender", []) + role_indices.get("Defender", [])

    for r in rucks:
        for m in mids:
            corr[r, m] = corr[m, r] = ruck_mids
    for m in mids:
        for f in forwards:
            corr[m, f] = corr[f, m] = mid_fwd
    for d in defenders:
        for f in forwards:
            corr[d, f] = corr[f, d] = def_fwd

    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def sample_correlated_uniforms(n_players: int, n_sims: int, corr: np.ndarray) -> np.ndarray:
    """Sample correlated uniform [0,1] values via Gaussian copula."""
    mean = np.zeros(n_players)
    z = np.random.multivariate_normal(mean, corr, size=n_sims)
    return norm.cdf(z)


def role_index_map(roles: list[str]) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    for i, role in enumerate(roles):
        indices.setdefault(role, []).append(i)
    return indices
