from hypothesis import (
    given,
    settings,
    strategies as st,
)
import numpy as np

from causal_validation.data import Dataset
from causal_validation.effects import (
    RandomEffect,
    StaticEffect,
)


@given(
    T=st.integers(min_value=2, max_value=50),
    N=st.integers(min_value=2, max_value=50),
    K=st.integers(min_value=1, max_value=10),
    effect=st.floats(
        min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=10)
def test_static_effect(T: int, N: int, K: int, effect: float, seed: int):
    rng = np.random.RandomState(seed)
    use_bernoulli = bool(rng.binomial(1, 0.5))
    include_X = bool(rng.binomial(1, 0.5))
    Y = rng.randn(T, N)
    D = rng.binomial(1, 0.3, (T, N)) if use_bernoulli else np.abs(rng.randn(T, N))
    X = rng.randn(T, N, K) if include_X else None

    data = Dataset(Y, D, X)
    static_effect = StaticEffect(effect=effect)

    inflated_data = static_effect(data)
    expected_inflation = np.ones(D.shape) + D * effect

    assert np.allclose(inflated_data.Y, Y * expected_inflation)
    assert np.allclose(inflated_data.D, D)
    if include_X:
        assert np.allclose(inflated_data.X, X)
    else:
        assert inflated_data.X is None


@given(
    N=st.integers(min_value=2, max_value=50),
    K=st.integers(min_value=1, max_value=10),
    mean_effect=st.floats(
        min_value=-0.3, max_value=0.3, allow_nan=False, allow_infinity=False
    ),
    stddev_effect=st.floats(
        min_value=0.01, max_value=0.2, allow_nan=False, allow_infinity=False
    ),
    seed=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=10)
def test_random_effect(
    N: int,
    K: int,
    mean_effect: float,
    stddev_effect: float,
    seed: int,
):
    rng = np.random.RandomState(seed)
    use_bernoulli = bool(rng.binomial(1, 0.5))
    include_X = bool(rng.binomial(1, 0.5))
    T = 50
    Y = rng.randn(T, N)
    D = rng.binomial(1, 0.3, (T, N)) if use_bernoulli else np.abs(rng.randn(T, N))
    X = rng.randn(T, N, K) if include_X else None

    data = Dataset(Y, D, X)
    random_effect = RandomEffect(mean_effect=mean_effect, stddev_effect=stddev_effect)

    inflated_data = random_effect(data, key=rng)

    assert inflated_data.Y.shape == Y.shape
    assert np.allclose(inflated_data.D, D)
    if include_X:
        assert np.allclose(inflated_data.X, X)
    else:
        assert inflated_data.X is None
