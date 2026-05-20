# Report Skeleton

## Project Title

Neural Network Regression for Bermudan Exercise Boundary: Longstaff-Schwartz Polynomial Regression vs Neural Network Regression.

## Research Objective

This project compares two continuation-value regression choices inside the standard Longstaff-Schwartz Monte Carlo backward induction method:

1. Classical polynomial regression.
2. Neural network regression.

The pricing and exercise algorithm itself remains the classical Longstaff-Schwartz method. The only planned change is the regression estimator used at each backward induction step.

## Planned Baseline

The first experiment will use a 1D Bermudan put option under the Black-Scholes model.

Planned workflow:

1. Simulate training paths.
2. Fit a Longstaff-Schwartz policy using polynomial regression.
3. Fit a Longstaff-Schwartz policy using neural network regression.
4. Evaluate both policies on independent test paths.
5. Compare prices, standard errors, runtimes, and exercise boundaries.

## Optional Extension

After the 1D put experiment, the project may add Bermudan max-call options for dimensions `d = 2, 5, 10`.

## Expected Outputs

- CSV result tables in `results/`.
- Figures in `figures/`.
- A concise comparison suitable for presentation.

## Current Status

Project scaffold only. No algorithms have been implemented yet.
