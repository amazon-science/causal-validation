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
from causal_validation.transforms import Noise, Trend
from causal_validation.transforms.parameter import TimeVaryingParameter

CONSTANTS = TestConstants()
DEFAULT_SEED = 123
GLOBAL_MEAN = 20
STATES = [42, 123]

def test_slot_type():
    noise_transform = Noise()
    assert isinstance(noise_transform.noise_dist, TimeVaryingParameter)

@given(loc=st.floats(min_value=-5., max_value=5.), 
       scale=st.floats(min_value=.1, max_value=1.))
@settings(max_examples=5)
def test_base_transform(loc: float, scale: float):
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)
    noise_transform = Noise(noise_dist = TimeVaryingParameter(sampling_dist=norm(loc, scale)))
    noisy_data = noise_transform(base_data)

    assert np.all(noisy_data.Xtr == base_data.Xtr)
    assert np.all(noisy_data.Xte == base_data.Xte)
    assert np.all(noisy_data.ytr != base_data.ytr)
    assert np.all(noisy_data.yte != base_data.yte)

@given(degree=st.integers(min_value=1, max_value=3), 
       coefficient=st.floats(min_value=-1., max_value=1.), 
       intercept=st.floats(min_value=-1., max_value=1.))
@settings(max_examples=5)
def test_composite_transform(degree: int, coefficient: float, intercept: float):
    trend_transform = Trend(degree=degree, coefficient=coefficient, intercept=intercept)
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)
    trendy_data = trend_transform(base_data)

    noise_transform = Noise()
    noisy_trendy_data = noise_transform(trendy_data)

    assert np.all(noisy_trendy_data.Xtr == trendy_data.Xtr)
    assert np.all(noisy_trendy_data.Xte == trendy_data.Xte)
    assert np.all(noisy_trendy_data.ytr != trendy_data.ytr)
    assert np.all(noisy_trendy_data.yte != trendy_data.yte)