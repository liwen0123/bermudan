"""Run the 1D Bermudan put Longstaff-Schwartz comparison."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ls_neural_network import LongstaffSchwartzNeuralNetwork
from src.ls_polynomial import LongstaffSchwartzPolynomial
from src.payoffs import bermudan_put_payoff
from src.simulation import simulate_gbm_paths


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 1D Bermudan put LS polynomial vs NN experiment."
    )
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--q", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-train-paths", type=int, default=20_000)
    parser.add_argument("--n-test-paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["polynomial", "neural_network"],
        default=["polynomial", "neural_network"],
        help="Methods to run. Use one or both.",
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Deprecated alias for --max-iter.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs is not None:
        args.max_iter = args.epochs
    return args


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Simulating independent train/test paths...")
    train_paths = simulate_gbm_paths(
        s0=args.S0,
        r=args.r,
        q=args.q,
        sigma=args.sigma,
        T=args.T,
        n_steps=args.n_steps,
        n_paths=args.n_train_paths,
        seed=args.seed,
    )
    test_paths = simulate_gbm_paths(
        s0=args.S0,
        r=args.r,
        q=args.q,
        sigma=args.sigma,
        T=args.T,
        n_steps=args.n_steps,
        n_paths=args.n_test_paths,
        seed=args.seed + 1,
    )

    payoff_fn = lambda state: bermudan_put_payoff(state, K=args.K)
    rows: list[dict[str, float | int | str]] = []
    fitted_policies: dict[str, object] = {}

    if "polynomial" in args.methods:
        print("Running LS polynomial regression...")
        policy = LongstaffSchwartzPolynomial(
            payoff_fn=payoff_fn,
            r=args.r,
            T=args.T,
            degree=args.degree,
        )
        result = policy.fit_evaluate(train_paths, test_paths)
        fitted_policies[f"Polynomial degree {args.degree}"] = policy
        rows.append(
            make_result_row(
                method=f"LS polynomial degree {args.degree}",
                result=result,
                args=args,
            )
        )

    if "neural_network" in args.methods:
        print("Running LS neural network regression...")
        policy = LongstaffSchwartzNeuralNetwork(
            payoff_fn=payoff_fn,
            r=args.r,
            T=args.T,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden_layers,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        result = policy.fit_evaluate(train_paths, test_paths)
        fitted_policies["Neural network"] = policy
        rows.append(
            make_result_row(
                method="LS neural network",
                result=result,
                args=args,
            )
        )

    results = pd.DataFrame(
        rows,
        columns=[
            "product",
            "dimension",
            "method",
            "price",
            "standard_error",
            "runtime_seconds",
            "n_train_paths",
            "n_test_paths",
            "seed",
        ],
    )
    results_path = RESULTS_DIR / "put_1d_results.csv"
    results.to_csv(results_path, index=False)

    save_price_comparison(results, FIGURES_DIR / "put_1d_price_comparison.png")
    save_boundary_plot(
        fitted_policies=fitted_policies,
        K=args.K,
        n_steps=args.n_steps,
        output_path=FIGURES_DIR / "put_1d_boundary.png",
    )

    print("\nSummary")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved results to {results_path}")
    print(f"Saved figures to {FIGURES_DIR}")


def make_result_row(method: str, result: object, args: argparse.Namespace) -> dict:
    return {
        "product": "1D Bermudan put",
        "dimension": 1,
        "method": method,
        "price": result.price,
        "standard_error": result.standard_error,
        "runtime_seconds": result.runtime_seconds,
        "n_train_paths": args.n_train_paths,
        "n_test_paths": args.n_test_paths,
        "seed": args.seed,
    }


def save_price_comparison(results: pd.DataFrame, output_path: Path) -> None:
    valid = results.dropna(subset=["price"])
    fig, ax = plt.subplots(figsize=(7, 4))

    if valid.empty:
        ax.text(0.5, 0.5, "No completed methods", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.bar(
            valid["method"],
            valid["price"],
            yerr=valid["standard_error"],
            color=["#4C78A8", "#F58518"][: len(valid)],
            capsize=4,
        )
        ax.set_ylabel("Estimated price")
        ax.set_title("1D Bermudan Put Price Comparison")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_boundary_plot(
    fitted_policies: dict[str, object],
    K: float,
    n_steps: int,
    output_path: Path,
) -> None:
    if not fitted_policies:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No fitted policies", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    steps = sorted(
        {
            max(1, min(n_steps - 1, int(round(frac * n_steps))))
            for frac in (0.25, 0.50, 0.75)
        }
    )
    policy_items = list(fitted_policies.items())
    x_grid = np.linspace(0.4 * K, 1.6 * K, 300).reshape(-1, 1)
    immediate = bermudan_put_payoff(x_grid, K=K).ravel()

    fig, axes = plt.subplots(
        nrows=len(steps),
        ncols=len(policy_items),
        figsize=(5 * len(policy_items), 3.2 * len(steps)),
        squeeze=False,
        sharex=True,
    )

    for row_idx, step in enumerate(steps):
        for col_idx, (label, policy) in enumerate(policy_items):
            ax = axes[row_idx, col_idx]
            continuation = predict_continuation(policy, step, x_grid)

            ax.plot(x_grid.ravel(), immediate, label="Immediate payoff", linewidth=2)
            if continuation is None:
                ax.text(0.5, 0.5, "No model at this date", ha="center", va="center")
            else:
                ax.plot(
                    x_grid.ravel(),
                    continuation,
                    label="Continuation value",
                    linewidth=2,
                )
                exercise = immediate >= continuation
                ax.fill_between(
                    x_grid.ravel(),
                    0.0,
                    np.maximum(immediate, continuation),
                    where=exercise,
                    alpha=0.18,
                    label="Exercise region",
                )
            ax.set_title(f"{label}, step {step}")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("Asset price")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3)
    fig.suptitle("1D Bermudan Put Exercise / Continue Regions", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def predict_continuation(policy: object, step: int, x_grid: np.ndarray) -> np.ndarray | None:
    model = getattr(policy, "models", {}).get(step)
    if model is None:
        return None

    if hasattr(model, "predict"):
        return np.asarray(model.predict(x_grid), dtype=float).ravel()

    predict_fn = getattr(policy, "_predict", None)
    if predict_fn is None:
        return None
    return np.asarray(predict_fn(model, x_grid), dtype=float).ravel()


if __name__ == "__main__":
    main()
