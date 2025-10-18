from hypothesis import (
    given,
    settings,
    strategies as st,
)
import numpy as np
from scipy.stats import norm

from causal_validation.testing import (
    TestConstants,
    simulate_data,
)
from causal_validation.transforms import Trend
from causal_validation.transforms.parameter import UnitVaryingParameter

CONSTANTS = TestConstants()
DEFAULT_SEED = 123
GLOBAL_MEAN = 20
STATES = [42, 123]


@st.composite
def coefficient_strategy(draw):
    lower_range = st.floats(
        min_value=-1,
        max_value=-1e-6,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    )
    upper_range = st.floats(
        min_value=1e-6,
        max_value=1,
        exclude_min=True,
        allow_infinity=False,
        allow_nan=False,
    )
    combined_strategy = st.one_of(lower_range, upper_range)
    return draw(combined_strategy)


@given(degree=st.integers(min_value=1, max_value=3), coefficient=coefficient_strategy())
@settings(max_examples=5)
def test_trend_coefficient(degree: int, coefficient: float):
    trend_transform = Trend(degree=degree, coefficient=coefficient, intercept=0)
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)
    data = trend_transform(base_data)

    if coefficient > 1:
        assert np.all(data.Y[-1, :] > base_data.Y[-1, :])
    elif coefficient < 0:
        assert np.all(data.Y[-1, :] < base_data.Y[-1, :])


@given(intercept=coefficient_strategy())
@settings(max_examples=5)
def test_trend_intercept(intercept: float):
    trend_transform = Trend(degree=1, coefficient=0, intercept=intercept)
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)
    data = trend_transform(base_data)

    if intercept > 0:
        assert np.all(data.Y > base_data.Y)
    elif intercept < 0:
        assert np.all(data.Y < base_data.Y)


@given(
    loc=st.floats(
        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=1e-3, max_value=10, allow_infinity=False, allow_nan=False
    ),
)
def test_varying_trend(loc: float, scale: float):
    data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)
    sampling_dist = norm(loc, scale)
    param = UnitVaryingParameter(sampling_dist=sampling_dist)
    trend = Trend(degree=1, coefficient=0.0, intercept=param)
    transformed_data = trend(data)
    assert not np.array_equal(transformed_data.Y[:, 0], transformed_data.Y[:, 1])

    trend = Trend(degree=1, coefficient=param, intercept=0.0)
    transformed_data = trend(data)
    assert not np.array_equal(transformed_data.Y[:, 0], transformed_data.Y[:, 1])


@given(
    loc=st.floats(
        min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    scale=st.floats(
        min_value=1e-3, max_value=10, allow_infinity=False, allow_nan=False
    ),
)
@settings(max_examples=5)
def test_randomness(loc: float, scale: float):
    data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)

    transformed_datas = []
    for random_state in STATES:
        sampling_dist = norm(loc, scale)
        param = UnitVaryingParameter(
            sampling_dist=sampling_dist, random_state=random_state
        )
        trend = Trend(degree=1, coefficient=0.0, intercept=param)
        transformed_data = trend(data)
        transformed_datas.append(transformed_data)

        assert np.array_equal(data.X, transformed_data.X)
        assert np.array_equal(data.D, transformed_data.D)
        assert not np.array_equal(data.Y, transformed_data.Y)


    assert np.array_equal(transformed_datas[0].X, transformed_datas[1].X)
    assert np.array_equal(transformed_datas[0].D, transformed_datas[1].D)
    assert not np.array_equal(transformed_datas[0].Y, transformed_datas[1].Y)
