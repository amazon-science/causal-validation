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
from causal_validation.transforms import (
    Noise,
    Trend,
)
from causal_validation.transforms.parameter import TimeVaryingParameter

CONSTANTS = TestConstants()
DEFAULT_SEED = 123
GLOBAL_MEAN = 20
STATES = [42, 123]


def test_slot_type():
    noise_transform = Noise()
    assert isinstance(noise_transform.noise_dist, TimeVaryingParameter)


def test_timepoints_randomness():
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)

    noise_transform = Noise()
    noisy_data = noise_transform(base_data)

    diff_tr = (noisy_data.ytr - base_data.ytr).reshape(-1)
    diff_te = (noisy_data.yte - base_data.yte).reshape(-1)

    assert np.all(diff_tr != diff_te)

    diff_tr_permute = np.random.permutation(diff_tr)
    diff_te_permute = np.random.permutation(diff_te)

    assert not np.all(diff_tr == diff_tr_permute)
    assert not np.all(diff_te == diff_te_permute)


@given(
    loc=st.floats(min_value=-5.0, max_value=5.0),
    scale=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=5)
def test_base_transform(loc: float, scale: float):
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)
    noise_transform = Noise(
        noise_dist=TimeVaryingParameter(sampling_dist=norm(loc, scale))
    )
    noisy_data = noise_transform(base_data)

    assert np.all(noisy_data.Xtr == base_data.Xtr)
    assert np.all(noisy_data.Xte == base_data.Xte)
    assert np.all(noisy_data.ytr != base_data.ytr)
    assert np.all(noisy_data.yte != base_data.yte)


@given(
    degree=st.integers(min_value=1, max_value=3),
    coefficient=st.floats(min_value=-1.0, max_value=1.0),
    intercept=st.floats(min_value=-1.0, max_value=1.0),
)
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


@given(
    loc_large=st.floats(min_value=10.0, max_value=15.0),
    loc_small=st.floats(min_value=-2.5, max_value=2.5),
    scale_large=st.floats(min_value=10.0, max_value=15.0),
    scale_small=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=5)
def test_perturbation_impact(
    loc_large: float, loc_small: float, scale_large: float, scale_small: float
):
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)

    noise_transform1 = Noise(
        noise_dist=TimeVaryingParameter(sampling_dist=norm(loc_small, scale_small))
    )
    noise_transform2 = Noise(
        noise_dist=TimeVaryingParameter(sampling_dist=norm(loc_small, scale_large))
    )
    noise_transform3 = Noise(
        noise_dist=TimeVaryingParameter(sampling_dist=norm(loc_large, scale_small))
    )

    noise_transforms = [noise_transform1, noise_transform2, noise_transform3]

    diff_tr_list, diff_te_list = [], []

    for noise_transform in noise_transforms:
        noisy_data = noise_transform(base_data)
        diff_tr = noisy_data.ytr - base_data.ytr
        diff_te = noisy_data.yte - base_data.yte
        diff_tr_list.append(diff_tr)
        diff_te_list.append(diff_te)

    assert np.max(diff_tr_list[0]) < np.max(diff_tr_list[1])
    assert np.min(diff_tr_list[0]) > np.min(diff_tr_list[1])
    assert np.max(diff_tr_list[0]) < np.max(diff_tr_list[2])
    assert np.min(diff_tr_list[0]) < np.min(diff_tr_list[2])

    assert np.max(diff_te_list[0]) < np.max(diff_te_list[1])
    assert np.min(diff_te_list[0]) > np.min(diff_te_list[1])
    assert np.max(diff_te_list[0]) < np.max(diff_te_list[2])
    assert np.min(diff_te_list[0]) < np.min(diff_te_list[2])
