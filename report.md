# Neural Network Regression for Bermudan Exercise Boundary

## Project Motivation

Bermudan option pricing is a natural setting for comparing regression models inside the Longstaff-Schwartz Monte Carlo framework. The exercise decision depends on an estimated continuation value, so the regression model can materially affect price quality, runtime, and the learned exercise boundary.

This project asks a simple question: if the Longstaff-Schwartz algorithm is kept fixed, how does a neural-network continuation model compare with classical polynomial regression?

## Bermudan Exercise Problem

At each exercise date, the holder chooses between immediate exercise and continuation. The pricing task is to estimate the continuation value from simulated paths and then work backward through time to determine the optimal exercise policy.

The pricing engine in this project is unchanged throughout. The only replacement is the continuation-value regressor:

1. Polynomial regression.
2. Neural-network regression using scikit-learn `MLPRegressor`.

## Longstaff-Schwartz Baseline

The baseline implementation follows the standard Longstaff-Schwartz backward induction method:

- simulate Monte Carlo paths under the Black-Scholes model,
- fit a regression model for continuation values at each exercise date,
- compare continuation with immediate exercise,
- roll backward through the exercise dates,
- evaluate the resulting policy on independent test paths.

This gives a direct comparison between polynomial regression and neural-network regression under the same simulation and exercise framework.

## Neural-Network Replacement Idea

The neural-network variant replaces the polynomial continuation fit with an `MLPRegressor`. This preserves the algorithmic structure while allowing a more flexible nonlinear approximation of the continuation value.

The tradeoff is practical:

- Neural networks may capture more complex state dependence.
- They are usually slower to train.
- They can require more tuning before convergence is reliable.

## Experimental Setup

### 1D Bermudan Put

- Black-Scholes model.
- Independent train and test paths.
- Polynomial regression degree 2.
- Neural-network continuation model with scikit-learn `MLPRegressor`.

Final 1D results:

| method | price | standard error | runtime seconds |
|---|---:|---:|---:|
| LS polynomial degree 2 | 6.027149 | 0.022837 | 0.209376 |
| LS neural network | 6.033014 | 0.022774 | 70.841067 |

### Bermudan Max-Call

Settings:

- dimensions = `2, 5, 10`
- `n_train_paths = 5000`
- `n_test_paths = 20000`
- `max_iter = 1000`
- neural network implemented with scikit-learn `MLPRegressor`

Final max-call results:

| dimension | method | price | standard error | runtime seconds |
|---:|---|---:|---:|---:|
| 2 | LS polynomial degree 2 | 33.936106 | 0.209772 | 0.044173 |
| 2 | LS neural network | 35.194152 | 0.233935 | 21.409312 |
| 5 | LS polynomial degree 2 | 57.904093 | 0.221222 | 0.057451 |
| 5 | LS neural network | 58.542651 | 0.235651 | 71.194198 |
| 10 | LS polynomial degree 2 | 76.175040 | 0.211657 | 0.213938 |
| 10 | LS neural network | 75.107426 | 0.222731 | 92.101903 |

## Interpretation

1. The Longstaff-Schwartz Monte Carlo backward induction framework is unchanged; only the continuation-value regression model is replaced.
2. The 1D Bermudan put benchmark is the main reproduction result. Neural-network regression gives a price very close to polynomial LS, so it is not dramatically worse when classical regression is accessible.
3. The Bermudan max-call experiment is a preliminary multi-dimensional validation for `d = 2, 5, 10`.
4. The neural-network max-call estimates are in the same range as polynomial LS, but they are not uniformly better.
5. `sklearn` `MLPRegressor` still hit the `1000`-iteration cap in every neural-network max-call run, so the max-call results should be interpreted cautiously.
6. In this small-scale setting, neural-network regression is much slower than polynomial LS.
7. The log-scale runtime figure is necessary because polynomial runtime is too small to compare directly with neural-network runtime.

## Limitations

- The max-call benchmark is still preliminary.
- The neural-network results depend on an `MLPRegressor` configuration that has not been fully tuned.
- The current experiments are intentionally small, so the conclusions are about relative behavior rather than final production pricing quality.

## Next Steps

- Run larger high-dimensional tests.
- Tune the neural-network architecture and optimization settings.
- Check whether convergence warnings disappear under a better training configuration.
- Compare boundary quality and stability across more exercise dates and more paths.

## Figures

- `figures/put_1d_price_comparison.png`
- `figures/put_1d_boundary.png`
- `figures/max_call_price_comparison.png`
- `figures/max_call_runtime.png`
- `figures/max_call_runtime_log.png`
