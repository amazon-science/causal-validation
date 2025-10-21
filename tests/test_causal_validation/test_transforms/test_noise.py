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
    TimeAndUnitVaryingParameter,
)

CONSTANTS = TestConstants()
DEFAULT_SEED = 123
GLOBAL_MEAN = 20
STATES = [42, 123]


def test_slot_type():
    noise_transform = Noise()
    assert isinstance(noise_transform.noise_dist, TimeAndUnitVaryingParameter)


def test_time_unit_points_randomness():
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)

    noise_transform = Noise()
    noisy_data = noise_transform(base_data)

    treated_unit_indices = base_data.treated_unit_indices
    ix_treat1 = treated_unit_indices[0]
    diff_treat1 = noisy_data.Y[:, ix_treat1] - base_data.Y[:, ix_treat1]

    ix_treat2 = treated_unit_indices[1]
    diff_treat2 = noisy_data.Y[:, ix_treat2] - base_data.Y[:, ix_treat2]

    assert np.all(diff_treat1 != diff_treat2)

    diff_treat1_1 = diff_treat1[: base_data.n_timepoints // 2]
    diff_treat1_2 = diff_treat1[
        base_data.n_timepoints // 2 : 2 * base_data.n_timepoints // 2
    ]

    assert np.all(diff_treat1_1 != diff_treat1_2)


@given(
    loc=st.floats(min_value=-5.0, max_value=5.0),
    scale=st.floats(min_value=0.1, max_value=1.0),
)
@settings(max_examples=5)
def test_base_transform(loc: float, scale: float):
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED)
    noise_transform = Noise(
        noise_dist=TimeAndUnitVaryingParameter(sampling_dist=norm(loc, scale))
    )
    noisy_data = noise_transform(base_data)
    treated_unit_indices = base_data.treated_unit_indices
    control_unit_indices = base_data.control_unit_indices

    assert np.all(noisy_data.X == base_data.X)
    assert np.all(noisy_data.D == base_data.D)
    assert np.all(
        noisy_data.Y[:, treated_unit_indices] != base_data.Y[:, treated_unit_indices]
    )
    assert np.all(
        noisy_data.Y[:, control_unit_indices] == base_data.Y[:, control_unit_indices]
    )


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

    treated_unit_indices = base_data.treated_unit_indices

    assert np.all(noisy_trendy_data.X == trendy_data.X)
    assert np.all(noisy_trendy_data.D == trendy_data.D)
    assert np.all(
        noisy_trendy_data.Y[:, treated_unit_indices]
        != trendy_data.Y[:, treated_unit_indices]
    )


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
        noise_dist=TimeAndUnitVaryingParameter(
            sampling_dist=norm(loc_small, scale_small)
        )
    )
    noise_transform2 = Noise(
        noise_dist=TimeAndUnitVaryingParameter(
            sampling_dist=norm(loc_small, scale_large)
        )
    )
    noise_transform3 = Noise(
        noise_dist=TimeAndUnitVaryingParameter(
            sampling_dist=norm(loc_large, scale_small)
        )
    )

    noise_transforms = [noise_transform1, noise_transform2, noise_transform3]

    treated_unit_indices = base_data.treated_unit_indices
    ix_treat = treated_unit_indices[0]

    diff_list = []

    for noise_transform in noise_transforms:
        noisy_data = noise_transform(base_data)
        diff = noisy_data.Y[:, ix_treat] - base_data.Y[:, ix_treat]
        diff_list.append(diff)

    assert np.max(diff_list[0]) < np.max(diff_list[1])
    assert np.min(diff_list[0]) > np.min(diff_list[1])
    assert np.max(diff_list[0]) < np.max(diff_list[2])
    assert np.min(diff_list[0]) < np.min(diff_list[2])


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

    assert np.all(noisy_data.Y == base_data.Y)
    assert np.all(noisy_data.D == base_data.D)
    assert np.all(noisy_data.X != base_data.X)


@given(n_covariates=st.integers(min_value=1, max_value=50))
@settings(max_examples=5)
def test_cov_composite_transform(n_covariates: int):
    CONSTANTS2 = TestConstants(N_COVARIATES=n_covariates)
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED, CONSTANTS2)

    covariate_noise_transform = CovariateNoise()
    cov_noisy_data = covariate_noise_transform(base_data)

    noise_transform = Noise()
    noisy_data = noise_transform(cov_noisy_data)

    assert np.all(noisy_data.X == cov_noisy_data.X)
    assert np.all(noisy_data.D == cov_noisy_data.D)

    treated_unit_indices = base_data.treated_unit_indices
    control_unit_indices = base_data.control_unit_indices
    assert np.all(
        noisy_data.Y[:, treated_unit_indices]
        != cov_noisy_data.Y[:, treated_unit_indices]
    )
    assert np.all(
        noisy_data.Y[:, control_unit_indices]
        == cov_noisy_data.Y[:, control_unit_indices]
    )


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

    diff_list = []

    for noise_transform in noise_transforms:
        noisy_data = noise_transform(base_data)
        diff = noisy_data.X - base_data.X
        diff_list.append(diff)

    assert np.max(diff_list[0]) < np.max(diff_list[1])
    assert np.min(diff_list[0]) > np.min(diff_list[1])
    assert np.max(diff_list[0]) < np.max(diff_list[2])
    assert np.min(diff_list[0]) < np.min(diff_list[2])
