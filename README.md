# Neural Network Regression for Bermudan Exercise Boundary

This project compares two continuation-value regression choices inside the Longstaff-Schwartz Monte Carlo backward induction framework:

1. Polynomial regression.
2. Neural network regression implemented with scikit-learn `MLPRegressor`.

The pricing algorithm itself is unchanged. Only the regression model used to estimate continuation values is replaced.

## Project Objective

The main goal is to reproduce a 1D Bermudan put benchmark and then extend the comparison to a multi-asset Bermudan max-call benchmark. The focus is on price, standard error, runtime, and the practical behavior of the learned exercise policy.

## Installation

Use Python 3.10 or later.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Torch is not required. The neural-network regressor uses scikit-learn `MLPRegressor`.

## Commands

Run tests:

```bash
pytest
```

Run the 1D Bermudan put experiment:

```bash
python scripts/run_1d_put_experiment.py
```

Run the Bermudan max-call experiment:

```bash
python scripts/run_max_call_experiment.py --dimensions 2 5 10 --n-train-paths 5000 --n-test-paths 20000 --max-iter 1000
```

## Output Files

Results are written to `results/`:

- `results/put_1d_results.csv`
- `results/max_call_results.csv`

Figures are written to `figures/`:

- `figures/put_1d_price_comparison.png`
- `figures/put_1d_boundary.png`
- `figures/max_call_price_comparison.png`
- `figures/max_call_runtime.png`
- `figures/max_call_runtime_log.png`

## Experimental Summary

Final 1D Bermudan put results:

| method | price | standard error | runtime seconds |
|---|---:|---:|---:|
| LS polynomial degree 2 | 6.027149 | 0.022837 | 0.209376 |
| LS neural network | 6.033014 | 0.022774 | 70.841067 |

Final Bermudan max-call results:

| dimension | method | price | standard error | runtime seconds |
|---:|---|---:|---:|---:|
| 2 | LS polynomial degree 2 | 33.936106 | 0.209772 | 0.044173 |
| 2 | LS neural network | 35.194152 | 0.233935 | 21.409312 |
| 5 | LS polynomial degree 2 | 57.904093 | 0.221222 | 0.057451 |
| 5 | LS neural network | 58.542651 | 0.235651 | 71.194198 |
| 10 | LS polynomial degree 2 | 76.175040 | 0.211657 | 0.213938 |
| 10 | LS neural network | 75.107426 | 0.222731 | 92.101903 |

## Notes

- The project keeps the Longstaff-Schwartz backward induction framework unchanged and only replaces the continuation-value regression model.
- The 1D Bermudan put benchmark is the main reproduction result.
- The Bermudan max-call benchmark is a preliminary multi-dimensional validation for `d = 2, 5, 10`.
- In the current small-scale setting, neural-network regression is much slower than polynomial LS.
- The neural-network max-call runs hit the `MLPRegressor` `1000`-iteration cap, so those results should be interpreted cautiously.
