from hypothesis import (
    given,
    strategies as st,
)
import numpy as np

from causal_validation.config import Config
from causal_validation.simulate import simulate


@given(
    n_units=st.integers(min_value=1, max_value=5),
    n_pre=st.integers(min_value=1, max_value=10),
    n_post=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_basic(n_units, n_pre, n_post, seed):
    cfg = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        seed=seed,
    )
    data = simulate(cfg)

    assert data.Xtr.shape == (n_pre, n_units)
    assert data.Xte.shape == (n_post, n_units)
    assert data.ytr.shape == (n_pre, 1)
    assert data.yte.shape == (n_post, 1)
    assert not data.has_covariates


@given(
    n_units=st.integers(min_value=1, max_value=5),
    n_pre=st.integers(min_value=1, max_value=10),
    n_post=st.integers(min_value=1, max_value=10),
    n_covariates=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_with_covariates(n_units, n_pre, n_post, n_covariates, seed):
    cfg = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        n_covariates=n_covariates,
        seed=seed,
    )
    data = simulate(cfg)

    assert data.Xtr.shape == (n_pre, n_units)
    assert data.Xte.shape == (n_post, n_units)
    assert data.ytr.shape == (n_pre, 1)
    assert data.yte.shape == (n_post, 1)
    assert data.has_covariates
    assert data.Ptr.shape == (n_pre, n_units, n_covariates)
    assert data.Pte.shape == (n_post, n_units, n_covariates)
    assert data.Rtr.shape == (n_pre, 1, n_covariates)
    assert data.Rte.shape == (n_post, 1, n_covariates)


@given(
    n_units=st.integers(min_value=1, max_value=5),
    n_pre=st.integers(min_value=1, max_value=10),
    n_post=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_reproducible(n_units, n_pre, n_post, seed):
    cfg1 = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        seed=seed,
    )
    data1 = simulate(cfg1)

    cfg2 = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        seed=seed,
    )
    data2 = simulate(cfg2)

    np.testing.assert_array_equal(data1.Xtr, data2.Xtr)
    np.testing.assert_array_equal(data1.Xte, data2.Xte)
    np.testing.assert_array_equal(data1.ytr, data2.ytr)
    np.testing.assert_array_equal(data1.yte, data2.yte)


@given(
    n_units=st.integers(min_value=1, max_value=3),
    n_pre=st.integers(min_value=3, max_value=10),
    n_post=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_covariate_effects(n_units, n_pre, n_post, seed):
    cfg = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        n_covariates=1,
        covariate_coeffs=np.array([10.0]),
        seed=seed,
    )
    data_with_cov = simulate(cfg)

    cfg_no_cov = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        seed=seed,
    )
    data_no_cov = simulate(cfg_no_cov)

    assert not np.allclose(data_with_cov.ytr, data_no_cov.ytr)
    assert not np.allclose(data_with_cov.yte, data_no_cov.yte)


@given(
    n_units=st.integers(min_value=1, max_value=3),
    n_pre=st.integers(min_value=3, max_value=10),
    n_post=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=1, max_value=1000),
)
def test_simulate_exact_covariate_effects(n_units, n_pre, n_post, seed):
    cfg = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        n_covariates=2,
        covariate_means=np.ones((n_units, 2)),
        covariate_stds=1e-12 * np.ones((n_units, 2)),
        covariate_coeffs=np.array([10.0, 5.0]),
        seed=seed,
    )
    data_with_cov = simulate(cfg)

    cfg_no_cov = Config(
        n_control_units=n_units,
        n_pre_intervention_timepoints=n_pre,
        n_post_intervention_timepoints=n_post,
        seed=seed,
    )
    data_no_cov = simulate(cfg_no_cov)

    assert np.allclose(data_with_cov.Xtr - 15, data_no_cov.Xtr)
    assert np.allclose(data_with_cov.Xte - 15, data_no_cov.Xte)
