# Neural Network Regression for Bermudan Exercise Boundary

Longstaff-Schwartz Polynomial Regression vs Neural Network Regression.

This repository is a focused quantitative finance reproduction project. The classical Longstaff-Schwartz Monte Carlo backward induction method will be kept unchanged; only the continuation-value regression step will be compared:

1. Polynomial regression.
2. Neural network regression implemented with scikit-learn `MLPRegressor`.

The primary target is the learned Bermudan exercise boundary / exercise frontier.

## Scope

First planned experiment:

- 1D Bermudan put option under the Black-Scholes model.
- Classical LS polynomial regression with degree 2 by default.
- LS neural network regression using `sklearn.neural_network.MLPRegressor`.
- Independent train and test Monte Carlo paths.
- Out-of-sample policy evaluation.
- Result tables saved to `results/`.
- Figures saved to `figures/`.

Optional extension:

- Bermudan max-call options for `d = 2, 5, 10`.

Out of scope:

- Bermudan swaption models.
- Cheyette model.
- LGM model.
- Tensor Neural Networks.
- BSDE framework.
- DANN.
- Joint learning.

## Project Structure

```text
.
|-- README.md
|-- REFERENCES.md
|-- report.md
|-- requirements.txt
|-- src/
|   |-- __init__.py
|   |-- simulation.py
|   |-- payoffs.py
|   |-- ls_polynomial.py
|   |-- ls_neural_network.py
|   |-- evaluation.py
|   `-- plotting.py
|-- scripts/
|   |-- run_1d_put_experiment.py
|   `-- run_max_call_experiment.py
|-- results/
|-- figures/
`-- tests/
    `-- test_basic.py
```

## Setup

Use Python 3.10 or later.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Commands

Run tests:

```bash
pytest
```

Run the 1D Bermudan put experiment:

```bash
python scripts/run_1d_put_experiment.py
```

Fast smoke run:

```bash
python scripts/run_1d_put_experiment.py --n-train-paths 1000 --n-test-paths 3000 --n-steps 10 --max-iter 20
```

Run the optional Bermudan max-call experiment:

```bash
python scripts/run_max_call_experiment.py
```

## Implementation Status

Implemented so far:

- Black-Scholes GBM path simulation.
- 1D Bermudan put and max-call payoff helpers.
- Classical Longstaff-Schwartz with polynomial regression.
- Longstaff-Schwartz with sklearn `MLPRegressor` continuation-value regression.
- 1D Bermudan put experiment script with CSV and figure outputs.
