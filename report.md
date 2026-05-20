# Report Skeleton

## Project Title

Neural Network Regression for Bermudan Exercise Boundary: Longstaff-Schwartz Polynomial Regression vs Neural Network Regression.

## Research Objective

This project compares two continuation-value regression choices inside the standard Longstaff-Schwartz Monte Carlo backward induction method:

1. Classical polynomial regression.
2. Neural network regression.

The pricing and exercise algorithm itself remains the classical Longstaff-Schwartz method. The only change is the regression estimator used at each backward induction step.

## Planned Baseline

The first experiment will use a 1D Bermudan put option under the Black-Scholes model.

Workflow:

1. Simulate training paths.
2. Fit a Longstaff-Schwartz policy using polynomial regression.
3. Fit a Longstaff-Schwartz policy using neural network regression.
4. Evaluate both policies on independent test paths.
5. Compare prices, standard errors, runtimes, and exercise boundaries.

The neural network continuation model is implemented with scikit-learn `MLPRegressor` inside a `Pipeline` with `StandardScaler`. This avoids a PyTorch dependency and keeps the project portable in restricted local environments.

Default neural network regression:

```text
hidden_layer_sizes = (64, 64)
activation = relu
solver = adam
max_iter = 300
```

## Optional Extension

After the 1D put experiment, the project may add Bermudan max-call options for dimensions `d = 2, 5, 10`.

## Expected Outputs

- CSV result tables in `results/`.
- Figures in `figures/`.
- A concise comparison suitable for presentation.

## Current Status

Implemented:

- GBM simulation under Black-Scholes.
- Payoff functions for 1D Bermudan put and multi-asset max-call.
- Polynomial Longstaff-Schwartz regression.
- Neural-network Longstaff-Schwartz regression with sklearn `MLPRegressor`.
- 1D Bermudan put experiment script with CSV and figure outputs.

Latest max-call results:

- `d=2`: polynomial `33.936106`, neural network `35.194152`
- `d=5`: polynomial `57.904093`, neural network `58.542651`

The `MLPRegressor` still hit the `1000`-iteration cap, so the max-call result is preliminary.
