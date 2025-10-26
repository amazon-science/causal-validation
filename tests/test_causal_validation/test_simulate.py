from hypothesis import (
    given,
    strategies as st,
)
import numpy as np

from causal_validation.config import Config
from causal_validation.simulate import simulate


@given(
    n_units=st.integers(min_value=2, max_value=50),
    n_timepoints=st.integers(min_value=5, max_value=150),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_basic(n_units, n_timepoints, seed):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        seed=seed,
    )
    data = simulate(cfg)

    assert data.Y.shape == (n_timepoints, n_units)
    assert data.D.shape == (n_timepoints, n_units)
    assert np.array_equal(data.D, treatment_assignments)
    assert data.X is None
    assert data.n_covariates == 0


@given(
    n_units=st.integers(min_value=2, max_value=50),
    n_timepoints=st.integers(min_value=5, max_value=100),
    n_covariates=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_with_covariates(n_units, n_timepoints, n_covariates, seed):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        n_covariates=n_covariates,
        seed=seed,
    )
    data = simulate(cfg)

    assert data.Y.shape == (n_timepoints, n_units)
    assert data.D.shape == (n_timepoints, n_units)
    assert data.X.shape == (n_timepoints, n_units, n_covariates)
    assert data.n_covariates == n_covariates


@given(
    n_units=st.integers(min_value=2, max_value=5),
    n_timepoints=st.integers(min_value=5, max_value=10),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_reproducible(n_units, n_timepoints, seed):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg1 = Config(
        treatment_assignments=treatment_assignments,
        seed=seed,
    )
    data1 = simulate(cfg1)

    cfg2 = Config(
        treatment_assignments=treatment_assignments,
        seed=seed,
    )
    data2 = simulate(cfg2)

    np.testing.assert_array_equal(data1.Y, data2.Y)
    np.testing.assert_array_equal(data1.D, data2.D)


@given(
    n_units=st.integers(min_value=2, max_value=3),
    n_timepoints=st.integers(min_value=6, max_value=10),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_covariate_effects(n_units, n_timepoints, seed):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        n_covariates=1,
        covariate_coeffs=np.array([10.0]),
        seed=seed,
    )
    data_with_cov = simulate(cfg)

    cfg_no_cov = Config(
        treatment_assignments=treatment_assignments,
        seed=seed,
    )
    data_no_cov = simulate(cfg_no_cov)

    assert not np.allclose(data_with_cov.Y, data_no_cov.Y)


@given(
    n_units=st.integers(min_value=2, max_value=3),
    n_timepoints=st.integers(min_value=6, max_value=10),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_exact_covariate_effects(n_units, n_timepoints, seed):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        n_covariates=2,
        covariate_means=np.ones(2),
        covariate_stds=1e-12 * np.ones(2),
        covariate_coeffs=np.array([10.0, 5.0]),
        seed=seed,
    )
    data_with_cov = simulate(cfg)

    cfg_no_cov = Config(
        treatment_assignments=treatment_assignments,
        seed=seed,
    )
    data_no_cov = simulate(cfg_no_cov)

    assert np.allclose(data_with_cov.Y - 15, data_no_cov.Y)


@given(
    n_units=st.integers(min_value=3, max_value=50),
    n_timepoints=st.integers(min_value=5, max_value=100),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_control_weighted(n_units, n_timepoints, seed):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        treated_simulation_type="control-weighted",
        seed=seed,
    )
    data = simulate(cfg)

    assert data.Y.shape == (n_timepoints, n_units)
    assert data.n_control_units == n_units - 1
    assert data.n_treated_units == 1
    assert np.all(data.Y[:, :-1] @ cfg.weights[0] == data.Y[:, -1])

    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -2:] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        treated_simulation_type="control-weighted",
        seed=seed,
    )
    data = simulate(cfg)

    N_TREATED = 2
    assert data.Y.shape == (n_timepoints, n_units)
    assert data.n_control_units == n_units - N_TREATED
    assert data.n_treated_units == N_TREATED
    assert np.all(data.Y[:, :-2] @ cfg.weights[0] == data.Y[:, -2])
    assert np.all(data.Y[:, :-2] @ cfg.weights[1] == data.Y[:, -1])


@given(
    n_units=st.integers(min_value=3, max_value=50),
    n_timepoints=st.integers(min_value=5, max_value=100),
    n_covariates=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_control_weighted_with_covariates(
    n_units, n_timepoints, n_covariates, seed
):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        treated_simulation_type="control-weighted",
        n_covariates=n_covariates,
        seed=seed,
    )
    data = simulate(cfg)

    assert data.Y.shape == (n_timepoints, n_units)
    assert data.n_control_units == n_units - 1
    assert data.n_treated_units == 1
    assert np.all(data.Y[:, :-1] @ cfg.weights[0] == data.Y[:, -1])
    X_treated = np.einsum("ijk,j->ik", data.X[:, :-1, :], cfg.weights[0])
    assert np.all(X_treated == data.X[:, -1, :])

    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -2:] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        treated_simulation_type="control-weighted",
        n_covariates=n_covariates,
        seed=seed,
    )
    data = simulate(cfg)

    N_TREATED = 2
    assert data.Y.shape == (n_timepoints, n_units)
    assert data.n_control_units == n_units - N_TREATED
    assert data.n_treated_units == N_TREATED
    assert np.all(data.Y[:, :-2] @ cfg.weights[0] == data.Y[:, -2])
    assert np.all(data.Y[:, :-2] @ cfg.weights[1] == data.Y[:, -1])
    X_treated1 = np.einsum("ijk,j->ik", data.X[:, :-2, :], cfg.weights[0])
    X_treated2 = np.einsum("ijk,j->ik", data.X[:, :-2, :], cfg.weights[1])
    assert np.all(X_treated1 == data.X[:, -2, :])
    assert np.all(X_treated2 == data.X[:, -1, :])


@given(
    n_units=st.integers(min_value=3, max_value=5000),
    n_timepoints=st.integers(min_value=5, max_value=10),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_independent(n_units, n_timepoints, seed):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        treated_simulation_type="independent",
        seed=seed,
    )
    data = simulate(cfg)

    assert data.Y.shape == (n_timepoints, n_units)
    assert data.n_control_units == n_units - 1
    assert data.n_treated_units == 1
