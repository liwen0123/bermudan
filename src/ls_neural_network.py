"""Neural-network Longstaff-Schwartz regression for Bermudan options.

This module uses scikit-learn's ``MLPRegressor`` for portability. The
Longstaff-Schwartz backward induction and out-of-sample evaluation logic match
the polynomial implementation; only the continuation-value regressor changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PayoffFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class LSNeuralNetworkEvaluation:
    """Out-of-sample evaluation result for a fitted neural LS policy."""

    price: float
    standard_error: float
    runtime_seconds: float
    exercise_decisions: np.ndarray | None = None


class LongstaffSchwartzNeuralNetwork:
    """Longstaff-Schwartz with sklearn MLP continuation regressions."""

    def __init__(
        self,
        payoff_fn: PayoffFunction,
        r: float,
        T: float,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        max_iter: int = 300,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        seed: int | None = None,
        random_state: int | None = None,
        epochs: int | None = None,
        device: str | None = None,
    ) -> None:
        """Create a neural-network Longstaff-Schwartz estimator.

        ``epochs`` and ``device`` are accepted for compatibility with the
        previous PyTorch-oriented API. If ``epochs`` is supplied, it overrides
        ``max_iter``. ``device`` is ignored because sklearn runs on CPU.
        """

        if T <= 0:
            raise ValueError("T must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be at least 1")
        if epochs is not None:
            max_iter = epochs
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.payoff_fn = payoff_fn
        self.r = r
        self.T = T
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.random_state = seed if random_state is None else random_state
        self.device = device
        self.models: dict[int, Pipeline] = {}
        self.fit_runtime_seconds: float | None = None
        self.n_steps_: int | None = None
        self.dim_: int | None = None

    def fit(self, paths: np.ndarray) -> "LongstaffSchwartzNeuralNetwork":
        """Fit one MLP continuation-value model per exercise date."""

        start = time.perf_counter()
        paths = self._validate_paths(paths)
        n_paths, n_dates, dim = paths.shape
        n_steps = n_dates - 1
        dt = self.T / n_steps
        discount = np.exp(-self.r * dt)

        self.models = {}
        self.n_steps_ = n_steps
        self.dim_ = dim

        cashflows = self._payoff(paths[:, -1, :])
        exercise_times = np.full(n_paths, n_steps, dtype=int)

        for step in range(n_steps - 1, 0, -1):
            state = paths[:, step, :]
            immediate = self._payoff(state)
            in_the_money = immediate > 0.0

            if not np.any(in_the_money):
                continue

            x_train = state[in_the_money]
            y_train = cashflows[in_the_money] * discount ** (
                exercise_times[in_the_money] - step
            )

            model = self._make_model(step)
            model.fit(x_train, y_train)
            self.models[step] = model

            continuation = model.predict(state)
            exercise = in_the_money & (immediate >= continuation)

            cashflows[exercise] = immediate[exercise]
            exercise_times[exercise] = step

        self.fit_runtime_seconds = time.perf_counter() - start
        return self

    def evaluate(
        self,
        paths: np.ndarray,
        return_exercise_decisions: bool = False,
    ) -> LSNeuralNetworkEvaluation:
        """Evaluate the fitted stopping policy on independent paths."""

        if self.n_steps_ is None:
            raise RuntimeError("fit must be called before evaluate")

        start = time.perf_counter()
        paths = self._validate_paths(paths)
        n_paths, n_dates, dim = paths.shape
        n_steps = n_dates - 1

        if n_steps != self.n_steps_:
            raise ValueError("test paths must have the same number of steps as training")
        if dim != self.dim_:
            raise ValueError("test paths must have the same dimension as training")

        dt = self.T / n_steps
        discount = np.exp(-self.r * dt)
        exercise_times = np.full(n_paths, n_steps, dtype=int)
        cashflows = self._payoff(paths[:, -1, :])
        active = np.ones(n_paths, dtype=bool)
        exercise_decisions = (
            np.zeros((n_paths, n_dates), dtype=bool)
            if return_exercise_decisions
            else None
        )

        for step in range(1, n_steps):
            model = self.models.get(step)
            if model is None:
                continue

            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break

            state = paths[active_idx, step, :]
            immediate = self._payoff(state)
            continuation = model.predict(state)
            exercise_local = immediate > 0.0
            exercise_local &= immediate >= continuation

            if not np.any(exercise_local):
                continue

            exercised_idx = active_idx[exercise_local]
            cashflows[exercised_idx] = immediate[exercise_local]
            exercise_times[exercised_idx] = step
            active[exercised_idx] = False

            if exercise_decisions is not None:
                exercise_decisions[exercised_idx, step] = True

        if exercise_decisions is not None:
            remaining_idx = np.flatnonzero(active)
            exercise_decisions[remaining_idx, n_steps] = True

        discounted_cashflows = cashflows * discount**exercise_times
        price = float(np.mean(discounted_cashflows))
        standard_error = float(
            np.std(discounted_cashflows, ddof=1) / np.sqrt(n_paths)
        )
        runtime = time.perf_counter() - start

        return LSNeuralNetworkEvaluation(
            price=price,
            standard_error=standard_error,
            runtime_seconds=runtime,
            exercise_decisions=exercise_decisions,
        )

    def fit_evaluate(
        self,
        train_paths: np.ndarray,
        test_paths: np.ndarray,
        return_exercise_decisions: bool = False,
    ) -> LSNeuralNetworkEvaluation:
        """Fit on training paths and evaluate on independent test paths."""

        start = time.perf_counter()
        self.fit(train_paths)
        result = self.evaluate(
            test_paths,
            return_exercise_decisions=return_exercise_decisions,
        )
        result.runtime_seconds = time.perf_counter() - start
        return result

    def _make_model(self, step: int) -> Pipeline:
        random_state = None if self.random_state is None else self.random_state + step
        hidden_layer_sizes = tuple([self.hidden_dim] * self.n_hidden_layers)
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=random_state,
        )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("mlp", mlp),
            ]
        )

    def _payoff(self, state: np.ndarray) -> np.ndarray:
        payoff = np.asarray(self.payoff_fn(state), dtype=float)
        return np.ravel(payoff)

    @staticmethod
    def _validate_paths(paths: np.ndarray) -> np.ndarray:
        paths = np.asarray(paths, dtype=float)
        if paths.ndim != 3:
            raise ValueError("paths must have shape (n_paths, n_steps + 1, dim)")
        if paths.shape[0] <= 1:
            raise ValueError("at least two paths are required")
        if paths.shape[1] <= 1:
            raise ValueError("at least one time step is required")
        if paths.shape[2] <= 0:
            raise ValueError("path dimension must be positive")
        return paths
