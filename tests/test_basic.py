import numpy as np

from src.ls_neural_network import LongstaffSchwartzNeuralNetwork
from src.ls_polynomial import LongstaffSchwartzPolynomial
from src.payoffs import bermudan_put_payoff, max_call_payoff
from src.simulation import simulate_gbm_paths


def test_simulate_gbm_paths_shape_1d():
    paths = simulate_gbm_paths(
        s0=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        n_steps=12,
        n_paths=500,
        seed=42,
    )

    assert paths.shape == (500, 13, 1)


def test_simulate_gbm_paths_shape_multi_dimensional():
    paths = simulate_gbm_paths(
        s0=np.array([100.0, 95.0, 105.0]),
        r=0.05,
        q=0.0,
        sigma=np.array([0.2, 0.25, 0.3]),
        T=1.0,
        n_steps=10,
        n_paths=100,
        seed=7,
    )

    assert paths.shape == (100, 11, 3)


def test_payoffs_are_non_negative():
    put_values = bermudan_put_payoff(np.array([80.0, 100.0, 120.0]), K=100.0)
    max_call_values = max_call_payoff(
        np.array([[90.0, 95.0], [100.0, 115.0], [120.0, 110.0]]),
        K=100.0,
    )

    assert np.all(put_values >= 0.0)
    assert np.all(max_call_values >= 0.0)


def test_fixed_seed_reproduces_paths():
    kwargs = dict(
        s0=np.array([100.0, 110.0]),
        r=0.03,
        q=0.01,
        sigma=0.2,
        T=2.0,
        n_steps=8,
        n_paths=50,
        seed=123,
    )

    first = simulate_gbm_paths(**kwargs)
    second = simulate_gbm_paths(**kwargs)

    np.testing.assert_allclose(first, second)


def test_polynomial_lsm_evaluates_out_of_sample_policy():
    train_paths = simulate_gbm_paths(
        s0=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        n_steps=5,
        n_paths=300,
        seed=1,
    )
    test_paths = simulate_gbm_paths(
        s0=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        n_steps=5,
        n_paths=200,
        seed=2,
    )
    policy = LongstaffSchwartzPolynomial(
        payoff_fn=lambda state: bermudan_put_payoff(state, K=100.0),
        r=0.05,
        T=1.0,
        degree=2,
    )

    result = policy.fit_evaluate(
        train_paths,
        test_paths,
        return_exercise_decisions=True,
    )

    assert result.price >= 0.0
    assert result.standard_error >= 0.0
    assert result.runtime_seconds >= 0.0
    assert result.exercise_decisions.shape == (200, 6)
    assert np.all(result.exercise_decisions.sum(axis=1) == 1)


def test_neural_network_lsm_evaluates_out_of_sample_policy():
    train_paths = simulate_gbm_paths(
        s0=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        n_steps=4,
        n_paths=120,
        seed=3,
    )
    test_paths = simulate_gbm_paths(
        s0=100.0,
        r=0.05,
        q=0.0,
        sigma=0.2,
        T=1.0,
        n_steps=4,
        n_paths=80,
        seed=4,
    )
    policy = LongstaffSchwartzNeuralNetwork(
        payoff_fn=lambda state: bermudan_put_payoff(state, K=100.0),
        r=0.05,
        T=1.0,
        hidden_dim=8,
        n_hidden_layers=1,
        max_iter=5,
        batch_size=64,
        learning_rate=1e-3,
        seed=5,
    )

    result = policy.fit_evaluate(
        train_paths,
        test_paths,
        return_exercise_decisions=True,
    )

    assert result.price >= 0.0
    assert result.standard_error >= 0.0
    assert result.runtime_seconds >= 0.0
    assert result.exercise_decisions.shape == (80, 5)
    assert np.all(result.exercise_decisions.sum(axis=1) == 1)
