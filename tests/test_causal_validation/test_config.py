from hypothesis import (
    given,
    strategies as st,
)
import numpy as np

from causal_validation.config import Config


@given(
    n_units=st.integers(min_value=2, max_value=10),
    n_timepoints=st.integers(min_value=5, max_value=20),
)
def test_config_basic_initialization(n_units, n_timepoints):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(treatment_assignments=treatment_assignments)
    assert cfg.treatment_assignments.shape == (n_timepoints, n_units)
    assert cfg.n_covariates is None
    assert cfg.covariate_means is None
    assert cfg.covariate_stds is None
    assert cfg.covariate_coeffs is None
    assert cfg.treated_simulation_type == "control-weighted"
    assert cfg.dirichlet_concentration == 1.0


@given(
    n_units=st.integers(min_value=2, max_value=5),
    n_timepoints=st.integers(min_value=5, max_value=10),
    n_covariates=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_config_with_covariates_auto_generation(
    n_units, n_timepoints, n_covariates, seed
):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -1] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        n_covariates=n_covariates,
        seed=seed,
    )
    assert cfg.n_covariates == n_covariates
    assert cfg.covariate_means.shape == (n_covariates,)
    assert cfg.covariate_stds.shape == (n_covariates,)
    assert np.all(cfg.covariate_stds > 0)
    assert cfg.covariate_coeffs.shape == (n_covariates,)
    assert np.all(cfg.covariate_stds >= 0)


@given(
    n_units=st.integers(min_value=2, max_value=3),
    n_covariates=st.integers(min_value=1, max_value=3),
)
def test_config_with_explicit_covariate_means(n_units, n_covariates):
    treatment_assignments = np.zeros((10, n_units))
    treatment_assignments[5:, -1] = 1
    means = np.random.random(n_covariates)

    cfg = Config(
        treatment_assignments=treatment_assignments,
        n_covariates=n_covariates,
        covariate_means=means,
    )
    np.testing.assert_array_equal(cfg.covariate_means, means)


@given(
    n_units=st.integers(min_value=2, max_value=3),
    n_covariates=st.integers(min_value=1, max_value=3),
)
def test_config_with_explicit_covariate_stds(n_units, n_covariates):
    treatment_assignments = np.zeros((10, n_units))
    treatment_assignments[5:, -1] = 1
    stds = np.random.random(n_covariates) + 0.1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        n_covariates=n_covariates,
        covariate_stds=stds,
    )
    np.testing.assert_array_equal(cfg.covariate_stds, stds)


@given(
    n_units=st.integers(min_value=4, max_value=50),
    n_timepoints=st.integers(min_value=2, max_value=1000),
    n_treated=st.integers(min_value=1, max_value=3),
)
def test_config_control_weighted_weights_generation(n_units, n_timepoints, n_treated):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, -n_treated:] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        treated_simulation_type="control-weighted",
    )
    assert len(cfg.weights) == n_treated
    assert all(len(w) == n_units - n_treated for w in cfg.weights)
    assert all(np.allclose(np.sum(w), 1.0) for w in cfg.weights)


@given(
    n_units=st.integers(min_value=3, max_value=50000),
    n_timepoints=st.integers(min_value=2, max_value=100),
)
def test_config_independent_simulation_type(n_units, n_timepoints):
    treatment_assignments = np.zeros((n_timepoints, n_units))
    treatment_assignments[n_timepoints // 2 :, n_units // 2 :] = 1

    cfg = Config(
        treatment_assignments=treatment_assignments,
        treated_simulation_type="independent",
    )
    assert cfg.treated_simulation_type == "independent"
    assert cfg.weights is None
