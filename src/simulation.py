"""Simulation utilities for Black-Scholes Monte Carlo paths."""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    s0: float | np.ndarray,
    r: float,
    q: float,
    sigma: float | np.ndarray,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion paths under Black-Scholes dynamics.

    The process is

        dS_t = (r - q) S_t dt + sigma S_t dW_t.

    Parameters
    ----------
    s0:
        Initial asset value. A scalar gives a 1D model; an array gives a
        multi-asset model.
    r:
        Risk-free rate.
    q:
        Continuous dividend yield.
    sigma:
        Volatility. A scalar is shared across dimensions; an array supplies one
        volatility per asset.
    T:
        Time to maturity.
    n_steps:
        Number of time steps.
    n_paths:
        Number of Monte Carlo paths.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated paths with shape ``(n_paths, n_steps + 1, dim)``.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if T <= 0:
        raise ValueError("T must be positive")

    s0_array = np.atleast_1d(np.asarray(s0, dtype=float))
    sigma_array = np.asarray(sigma, dtype=float)
    dim = s0_array.size

    if sigma_array.ndim == 0:
        sigma_array = np.full(dim, float(sigma_array))
    else:
        sigma_array = np.atleast_1d(sigma_array).astype(float)

    if sigma_array.shape != (dim,):
        raise ValueError("sigma must be a scalar or have the same length as s0")
    if np.any(s0_array <= 0):
        raise ValueError("s0 values must be positive")
    if np.any(sigma_array < 0):
        raise ValueError("sigma values must be non-negative")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    shocks = rng.normal(size=(n_paths, n_steps, dim))

    drift = (r - q - 0.5 * sigma_array**2) * dt
    diffusion = sigma_array * np.sqrt(dt) * shocks
    log_returns = drift + diffusion

    paths = np.empty((n_paths, n_steps + 1, dim), dtype=float)
    paths[:, 0, :] = s0_array
    paths[:, 1:, :] = s0_array * np.exp(np.cumsum(log_returns, axis=1))
    return paths
