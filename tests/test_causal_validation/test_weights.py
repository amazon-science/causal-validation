from hypothesis import (
    given,
    strategies as st,
)
import numpy as np

from causal_validation.weights import UniformWeights


@given(
    n_units=st.integers(min_value=1, max_value=100),
    n_time=st.integers(min_value=1, max_value=100),
)
def test_uniform_weights(n_units: int, n_time: int):
    weights = UniformWeights()
    data = np.random.random(size=(n_time, n_units))
    weight_vals = weights.get_weights(data)
    np.testing.assert_almost_equal(np.mean(weight_vals), weight_vals, decimal=6)
    assert weight_vals.shape == (n_units, 1)


@given(
    n_units=st.integers(min_value=1, max_value=100),
    n_time=st.integers(min_value=1, max_value=100),
)
def test_weight_contr(n_units: int, n_time: int):
    obs = np.ones(shape=(n_time, n_units))
    weighted_obs = UniformWeights()(obs)
    np.testing.assert_almost_equal(np.mean(weighted_obs), weighted_obs, decimal=6)
    np.testing.assert_almost_equal(
        obs @ UniformWeights().get_weights(obs), weighted_obs, decimal=6
    )


@given(
    n_units=st.integers(min_value=1, max_value=10),
    n_time=st.integers(min_value=1, max_value=10),
    n_covariates=st.integers(min_value=1, max_value=5),
)
def test_weight_contr_3d(n_units: int, n_time: int, n_covariates: int):
    covariates = np.ones(shape=(n_time, n_units, n_covariates))
    weights = UniformWeights()
    weighted_covs = weights.weight_contr(covariates)

    assert weighted_covs.shape == (n_time, 1, n_covariates)
    expected = np.einsum(
        "n d k, d i -> n i k", covariates, weights.get_weights(covariates)
    )
    np.testing.assert_almost_equal(weighted_covs, expected, decimal=6)


def test_weights_sum_to_one():
    obs = np.random.random((10, 5))
    weights = UniformWeights()
    weight_vals = weights.get_weights(obs)
    np.testing.assert_almost_equal(weight_vals.sum(), 1.0, decimal=6)


def test_weights_non_negative():
    obs = np.random.random((10, 5))
    weights = UniformWeights()
    weight_vals = weights.get_weights(obs)
    assert np.all(weight_vals >= 0)
