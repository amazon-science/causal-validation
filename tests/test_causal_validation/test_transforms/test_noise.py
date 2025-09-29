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
    CovariateNoise,
    Noise,
    Trend,
)
from causal_validation.transforms.parameter import (
    CovariateNoiseParameter,
    TimeVaryingParameter,
)

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


# Covariate Noise Test
def test_cov_slot_type():
    noise_transform = CovariateNoise()
    assert isinstance(noise_transform.noise_dist, CovariateNoiseParameter)


@given(n_covariates=st.integers(min_value=1, max_value=50))
@settings(max_examples=5)
def test_output_covariate_transform(n_covariates: int):
    CONSTANTS2 = TestConstants(N_COVARIATES=n_covariates)
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED, CONSTANTS2)

    covariate_noise_transform = CovariateNoise()
    noisy_data = covariate_noise_transform(base_data)

    assert np.all(noisy_data.ytr == base_data.ytr)
    assert np.all(noisy_data.yte == base_data.yte)
    assert np.all(noisy_data.Xtr == base_data.Xtr)
    assert np.all(noisy_data.Xte == base_data.Xte)

    diff_Ptr = (noisy_data.Ptr - base_data.Ptr).reshape(-1)
    diff_Pte = (noisy_data.Pte - base_data.Pte).reshape(-1)

    assert np.all(diff_Ptr != diff_Pte)

    diff_Rtr = (noisy_data.Rtr - base_data.Rtr).reshape(-1)
    diff_Rte = (noisy_data.Rte - base_data.Rte).reshape(-1)

    assert np.all(diff_Rtr != diff_Rte)

    diff_Ptr_permute = np.random.permutation(diff_Ptr)
    diff_Pte_permute = np.random.permutation(diff_Pte)
    diff_Rtr_permute = np.random.permutation(diff_Rtr)
    diff_Rte_permute = np.random.permutation(diff_Rte)

    assert not np.all(diff_Ptr == diff_Ptr_permute)
    assert not np.all(diff_Pte == diff_Pte_permute)
    assert not np.all(diff_Rtr == diff_Rtr_permute)
    assert not np.all(diff_Rte == diff_Rte_permute)


@given(n_covariates=st.integers(min_value=1, max_value=50))
@settings(max_examples=5)
def test_cov_composite_transform(n_covariates: int):
    CONSTANTS2 = TestConstants(N_COVARIATES=n_covariates)
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED, CONSTANTS2)

    covariate_noise_transform = CovariateNoise()
    cov_noisy_data = covariate_noise_transform(base_data)

    noise_transform = Noise()
    noisy_data = noise_transform(cov_noisy_data)

    assert np.all(noisy_data.Xtr == cov_noisy_data.Xtr)
    assert np.all(noisy_data.Xte == cov_noisy_data.Xte)
    assert np.all(noisy_data.Ptr == cov_noisy_data.Ptr)
    assert np.all(noisy_data.Pte == cov_noisy_data.Pte)
    assert np.all(noisy_data.Rtr == cov_noisy_data.Rtr)
    assert np.all(noisy_data.Rte == cov_noisy_data.Rte)
    assert np.all(noisy_data.ytr != cov_noisy_data.ytr)
    assert np.all(noisy_data.yte != cov_noisy_data.yte)


@given(
    loc_large=st.floats(min_value=10.0, max_value=15.0),
    loc_small=st.floats(min_value=-2.5, max_value=2.5),
    scale_large=st.floats(min_value=10.0, max_value=15.0),
    scale_small=st.floats(min_value=0.1, max_value=1.0),
    n_covariates=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=5)
def test_cov_perturbation_impact(
    loc_large: float,
    loc_small: float,
    scale_large: float,
    scale_small: float,
    n_covariates: int,
):
    CONSTANTS2 = TestConstants(N_COVARIATES=n_covariates)
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED, CONSTANTS2)

    noise_transform1 = CovariateNoise(
        noise_dist=CovariateNoiseParameter(sampling_dist=norm(loc_small, scale_small))
    )
    noise_transform2 = CovariateNoise(
        noise_dist=CovariateNoiseParameter(sampling_dist=norm(loc_small, scale_large))
    )
    noise_transform3 = CovariateNoise(
        noise_dist=CovariateNoiseParameter(sampling_dist=norm(loc_large, scale_small))
    )

    noise_transforms = [noise_transform1, noise_transform2, noise_transform3]

    diff_Rtr_list, diff_Rte_list = [], []
    diff_Ptr_list, diff_Pte_list = [], []

    for noise_transform in noise_transforms:
        noisy_data = noise_transform(base_data)
        diff_Rtr = noisy_data.Rtr - base_data.Rtr
        diff_Rte = noisy_data.Rte - base_data.Rte
        diff_Ptr = noisy_data.Ptr - base_data.Ptr
        diff_Pte = noisy_data.Pte - base_data.Pte
        diff_Rtr_list.append(diff_Rtr)
        diff_Rte_list.append(diff_Rte)
        diff_Ptr_list.append(diff_Ptr)
        diff_Pte_list.append(diff_Pte)

    assert np.max(diff_Rtr_list[0]) < np.max(diff_Rtr_list[1])
    assert np.min(diff_Rtr_list[0]) > np.min(diff_Rtr_list[1])
    assert np.max(diff_Rtr_list[0]) < np.max(diff_Rtr_list[2])
    assert np.min(diff_Rtr_list[0]) < np.min(diff_Rtr_list[2])

    assert np.max(diff_Rte_list[0]) < np.max(diff_Rte_list[1])
    assert np.min(diff_Rte_list[0]) > np.min(diff_Rte_list[1])
    assert np.max(diff_Rte_list[0]) < np.max(diff_Rte_list[2])
    assert np.min(diff_Rte_list[0]) < np.min(diff_Rte_list[2])

    assert np.max(diff_Ptr_list[0]) < np.max(diff_Ptr_list[1])
    assert np.min(diff_Ptr_list[0]) > np.min(diff_Ptr_list[1])
    assert np.max(diff_Ptr_list[0]) < np.max(diff_Ptr_list[2])
    assert np.min(diff_Ptr_list[0]) < np.min(diff_Ptr_list[2])

    assert np.max(diff_Pte_list[0]) < np.max(diff_Pte_list[1])
    assert np.min(diff_Pte_list[0]) > np.min(diff_Pte_list[1])
    assert np.max(diff_Pte_list[0]) < np.max(diff_Pte_list[2])
    assert np.min(diff_Pte_list[0]) < np.min(diff_Pte_list[2])
