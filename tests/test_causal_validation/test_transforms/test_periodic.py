from hypothesis import (
    given,
    settings,
    strategies as st,
)
import numpy as np
from scipy.stats import norm

from causal_validation.data import Dataset
from causal_validation.testing import (
    TestConstants,
    simulate_data,
)
from causal_validation.transforms import Periodic
from causal_validation.transforms.parameter import UnitVaryingParameter

CONSTANTS = TestConstants()
DEFAULT_SEED = 123
GLOBAL_MEAN = 20
GLOBAL_SCALE = 0.5


@given(
    frequency=st.integers(min_value=1, max_value=20),
    amplitude=st.floats(
        min_value=-100, max_value=100, allow_infinity=False, allow_nan=False
    ),
    shift=st.floats(
        min_value=-100, max_value=100, allow_infinity=False, allow_nan=False
    ),
    offset=st.floats(
        min_value=-100, max_value=100, allow_infinity=False, allow_nan=False
    ),
    global_mean=st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
@settings(max_examples=5)
def test_periodic_initialisation(
    frequency: int,
    amplitude: float,
    shift: float,
    offset: float,
    global_mean: float,
):
    periodic_transform = Periodic(
        amplitude=amplitude, frequency=frequency, shift=shift, offset=offset
    )
    base_data = simulate_data(global_mean, DEFAULT_SEED)
    data = periodic_transform(base_data)
    assert isinstance(data, Dataset)
    for slot in CONSTANTS.DATA_SLOTS:
        _base_data_array = getattr(base_data, slot)
        _data_array = getattr(data, slot)
        assert _base_data_array.shape == _data_array.shape
        assert np.sum(np.isnan(_base_data_array)) == 0
        assert np.sum(np.isnan(_data_array)) == 0


@given(
    frequency=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=1, max_value=30),
    global_mean=st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
def test_frequency_param(frequency: int, seed: int, global_mean: float):
    periodic_transform = Periodic(amplitude=1, frequency=frequency, shift=0, offset=0)
    base_data = simulate_data(global_mean, seed)
    data = periodic_transform(base_data)
    np.testing.assert_array_almost_equal(
        np.mean(data.Y, axis=0), np.mean(base_data.Y, axis=0)
    )


@st.composite
def amplitude_strategy(draw):
    lower_range = st.floats(
        min_value=-100,
        max_value=-1e-6,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    )
    upper_range = st.floats(
        min_value=1e-6,
        max_value=100,
        exclude_min=True,
        allow_infinity=False,
        allow_nan=False,
    )
    combined_strategy = st.one_of(lower_range, upper_range)
    return draw(combined_strategy)


@given(
    amplitude=amplitude_strategy(),
    seed=st.integers(min_value=1, max_value=30),
    global_mean=st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
def test_amplitude_param(amplitude: float, seed: int, global_mean: float):
    periodic_transform = Periodic(frequency=1, amplitude=amplitude, shift=0, offset=0)
    base_data = simulate_data(global_mean, seed)
    data = periodic_transform(base_data)

    assert np.isclose(np.max(data.Y - base_data.Y), np.abs(amplitude), rtol=1)


@given(
    frequency=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=1, max_value=30),
    global_mean=st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
def test_num_frequencies(frequency: int, seed: int, global_mean: float):
    constants = TestConstants(TREATMENT_ASSIGNMENTS=np.random.randn(100, 20))
    periodic_transform = Periodic(frequency=frequency, amplitude=1, shift=0, offset=0)
    base_data = simulate_data(global_mean, seed, constants=constants)
    data = periodic_transform(base_data)

    periodic_signal = data.Y - base_data.Y

    for i in range(data.n_units):
        signal = periodic_signal[:, i]
        num_samples = len(signal)

        fft_vals = np.fft.fft(signal)
        fft_magnitudes = np.abs(fft_vals)

        positive_freqs = fft_magnitudes[1 : num_samples // 2]
        peak_frequency = np.argmax(positive_freqs) + 1

        np.testing.assert_equal(peak_frequency, frequency)


@given(
    offset=st.floats(
        min_value=-100, max_value=100, allow_infinity=False, allow_nan=False
    ),
    seed=st.integers(min_value=1, max_value=30),
    global_mean=st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
def test_offset(offset: float, seed: int, global_mean: float):
    periodic_transform = Periodic(frequency=1, amplitude=5, shift=0, offset=offset)
    base_data = simulate_data(global_mean, seed)
    data = periodic_transform(base_data)

    mean_diff = np.mean(data.Y - base_data.Y)
    assert np.isclose(mean_diff, offset, atol=0.1)


def test_varying_parameters():
    periodic_transform = Periodic()
    param_slots = periodic_transform._slots
    constants = TestConstants()
    data_slots = constants.DATA_SLOTS
    base_data = simulate_data(GLOBAL_MEAN, DEFAULT_SEED, constants=constants)
    base_data_transform = periodic_transform(base_data)
    for i, slot in enumerate(param_slots):
        setattr(
            periodic_transform,
            slot,
            UnitVaryingParameter(sampling_dist=norm(GLOBAL_MEAN, GLOBAL_SCALE)),
        )
        data = periodic_transform(base_data)
        for dslot in data_slots:
            assert not np.any(np.isnan(getattr(data, dslot)))

        assert np.array_equal(data.X, base_data_transform.X)
        assert np.array_equal(data.D, base_data_transform.D)
        assert not np.array_equal(data.Y, base_data_transform.Y)
