"""Payoff functions for Bermudan option experiments."""

from __future__ import annotations

import numpy as np


def bermudan_put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    """Return the 1D Bermudan put payoff ``max(K - S, 0)``."""

    values = np.asarray(S, dtype=float)
    return np.maximum(K - values, 0.0)


def max_call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    """Return the multi-asset max-call payoff ``max(max_i S_i - K, 0)``."""

    values = np.asarray(S, dtype=float)
    max_assets = np.max(values, axis=-1)
    return np.maximum(max_assets - K, 0.0)
