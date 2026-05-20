"""Polynomial Longstaff-Schwartz regression for Bermudan options."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


PayoffFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class LSPolynomialEvaluation:
    """Out-of-sample evaluation result for a fitted LS policy."""

    price: float
    standard_error: float
    runtime_seconds: float
    exercise_decisions: np.ndarray | None = None


class LongstaffSchwartzPolynomial:
    """Classical Longstaff-Schwartz with polynomial continuation regression.

    The backward induction algorithm is the standard LS Monte Carlo method. The
    continuation value estimator at each exercise date is a scikit-learn
    polynomial regression pipeline.
    """

    def __init__(
        self,
        payoff_fn: PayoffFunction,
        r: float,
        T: float,
        degree: int = 2,
        regression: str = "linear",
        ridge_alpha: float = 1.0,
    ) -> None:
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if T <= 0:
            raise ValueError("T must be positive")
        if regression not in {"linear", "ridge"}:
            raise ValueError("regression must be either 'linear' or 'ridge'")

        self.payoff_fn = payoff_fn
        self.r = r
        self.T = T
        self.degree = degree
        self.regression = regression
        self.ridge_alpha = ridge_alpha
        self.models: dict[int, Pipeline] = {}
        self.fit_runtime_seconds: float | None = None
        self.n_steps_: int | None = None
        self.dim_: int | None = None

    def fit(self, paths: np.ndarray) -> "LongstaffSchwartzPolynomial":
        """Fit continuation-value regressions by backward induction."""

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

            model = self._make_model()
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
    ) -> LSPolynomialEvaluation:
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

        return LSPolynomialEvaluation(
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
    ) -> LSPolynomialEvaluation:
        """Fit on training paths and evaluate on independent test paths."""

        start = time.perf_counter()
        self.fit(train_paths)
        result = self.evaluate(
            test_paths,
            return_exercise_decisions=return_exercise_decisions,
        )
        result.runtime_seconds = time.perf_counter() - start
        return result

    def _make_model(self) -> Pipeline:
        if self.regression == "ridge":
            estimator = Ridge(alpha=self.ridge_alpha)
        else:
            estimator = LinearRegression()

        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=True)),
                ("regression", estimator),
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
