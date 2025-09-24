import numpy as np
from hypothesis import given, strategies as st

from causal_validation.config import Config


@given(
    n_units=st.integers(min_value=1, max_value=10),
    n_pre=st.integers(min_value=1, max_value=20),
    n_post=st.integers(min_value=1, max_value=20)
)
def test_config_basic_initialization(n_units, n_pre, n_post):
    cfg = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post
    )
    assert cfg.n_control_units == n_units
    assert cfg.n_pre_intervention_timepoints == n_pre
    assert cfg.n_post_intervention_timepoints == n_post
    assert cfg.n_covariates is None
    assert cfg.covariate_means is None
    assert cfg.covariate_stds is None
    assert cfg.covariate_coeffs is None


@given(
    n_units=st.integers(min_value=1, max_value=5),
    n_pre=st.integers(min_value=1, max_value=10),
    n_post=st.integers(min_value=1, max_value=10),
    n_covariates=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=1, max_value=1000)
)
def test_config_with_covariates_auto_generation(
    n_units, n_pre, n_post, n_covariates, seed
):
    cfg = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        n_covariates=n_covariates,
        seed=seed
    )
    assert cfg.n_covariates == n_covariates
    assert cfg.covariate_means.shape == (n_units, n_covariates)
    assert cfg.covariate_stds.shape == (n_units, n_covariates)
    assert cfg.covariate_coeffs.shape == (n_covariates,)
    assert np.all(cfg.covariate_stds >= 0)


@given(
    n_units=st.integers(min_value=1, max_value=3),
    n_covariates=st.integers(min_value=1, max_value=3)
)
def test_config_with_explicit_covariate_means(n_units, n_covariates):
    means = np.random.random((n_units, n_covariates))
    cfg = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=10,
        n_post_intervention_timepoints=5,
        n_covariates=n_covariates,
        covariate_means=means
    )
    np.testing.assert_array_equal(cfg.covariate_means, means)


@given(
    n_units=st.integers(min_value=1, max_value=3),
    n_covariates=st.integers(min_value=1, max_value=3)
)
def test_config_with_explicit_covariate_stds(n_units, n_covariates):
    stds = np.random.random((n_units, n_covariates)) + 0.1
    cfg = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=10,
        n_post_intervention_timepoints=5,
        n_covariates=n_covariates,
        covariate_stds=stds
    )
    np.testing.assert_array_equal(cfg.covariate_stds, stds)
