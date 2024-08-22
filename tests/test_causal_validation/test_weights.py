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
def test_weight_obs(n_units: int, n_time: int):
    obs = np.ones(shape=(n_time, n_units))
    weighted_obs = UniformWeights()(obs)
    np.testing.assert_almost_equal(np.mean(weighted_obs), weighted_obs, decimal=6)
    np.testing.assert_almost_equal(
        obs @ UniformWeights().get_weights(obs), weighted_obs, decimal=6
    )
