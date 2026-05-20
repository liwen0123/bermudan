"""Run a Bermudan max-call Longstaff-Schwartz smoke experiment."""

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
from src.payoffs import max_call_payoff
from src.simulation import simulate_gbm_paths


RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bermudan max-call LS polynomial vs NN experiments."
    )
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--S0", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--q", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=3.0)
    parser.add_argument("--n-steps", type=int, default=9)
    parser.add_argument("--n-train-paths", type=int, default=30_000)
    parser.add_argument("--n-test-paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["polynomial", "neural_network"],
        default=["polynomial", "neural_network"],
    )
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for dim in args.dimensions:
        print(f"\nDimension d={dim}")
        s0 = np.full(dim, args.S0)
        sigma = np.full(dim, args.sigma)
        train_seed = args.seed + 1000 * dim
        test_seed = train_seed + 1

        print("Simulating independent train/test paths...")
        train_paths = simulate_gbm_paths(
            s0=s0,
            r=args.r,
            q=args.q,
            sigma=sigma,
            T=args.T,
            n_steps=args.n_steps,
            n_paths=args.n_train_paths,
            seed=train_seed,
        )
        test_paths = simulate_gbm_paths(
            s0=s0,
            r=args.r,
            q=args.q,
            sigma=sigma,
            T=args.T,
            n_steps=args.n_steps,
            n_paths=args.n_test_paths,
            seed=test_seed,
        )

        payoff_fn = lambda state, strike=args.K: max_call_payoff(state, K=strike)

        if "polynomial" in args.methods:
            print("Running LS polynomial regression...")
            policy = LongstaffSchwartzPolynomial(
                payoff_fn=payoff_fn,
                r=args.r,
                T=args.T,
                degree=args.degree,
            )
            result = policy.fit_evaluate(train_paths, test_paths)
            rows.append(make_result_row(dim, f"LS polynomial degree {args.degree}", result, args))

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
                seed=args.seed + dim,
            )
            result = policy.fit_evaluate(train_paths, test_paths)
            rows.append(make_result_row(dim, "LS neural network", result, args))

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
    results_path = RESULTS_DIR / "max_call_results.csv"
    results.to_csv(results_path, index=False)
    save_price_comparison(results, FIGURES_DIR / "max_call_price_comparison.png")
    save_runtime_plot(results, FIGURES_DIR / "max_call_runtime.png")

    print("\nSummary")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved results to {results_path}")
    print(f"Saved figures to {FIGURES_DIR}")


def make_result_row(
    dim: int,
    method: str,
    result: object,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    return {
        "product": "Bermudan max-call",
        "dimension": dim,
        "method": method,
        "price": result.price,
        "standard_error": result.standard_error,
        "runtime_seconds": result.runtime_seconds,
        "n_train_paths": args.n_train_paths,
        "n_test_paths": args.n_test_paths,
        "seed": args.seed,
    }


def save_price_comparison(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, group in results.groupby("method"):
        group = group.sort_values("dimension")
        ax.errorbar(
            group["dimension"],
            group["price"],
            yerr=group["standard_error"],
            marker="o",
            capsize=4,
            label=method,
        )
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Estimated price")
    ax.set_title("Bermudan Max-Call Price Comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_runtime_plot(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, group in results.groupby("method"):
        group = group.sort_values("dimension")
        ax.plot(
            group["dimension"],
            group["runtime_seconds"],
            marker="o",
            label=method,
        )
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Runtime seconds")
    ax.set_title("Bermudan Max-Call Runtime")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
