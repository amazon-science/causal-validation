from copy import deepcopy
import string
import typing as tp

from azcausal.estimators.panel.did import DID
from hypothesis import (
    given,
    settings,
    strategies as st,
)
import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
import pytest

from causal_validation.data import (
    Dataset,
    DatasetContainer,
    reassign_treatment,
)
from causal_validation.testing import (
    TestConstants,
    simulate_data,
)
from causal_validation.types import InterventionTypes

MIN_STRING_LENGTH = 1
MAX_STRING_LENGTH = 20
DEFAULT_SEED = 123
NUM_NON_CONTROL_COLS = 2
LARGE_N_POST = 5000
LARGE_N_PRE = 5000


@given(
    seed=st.integers(min_value=1, max_value=30),
    global_mean=st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
def test_global_mean(seed: int, global_mean: float):
    constants = TestConstants(
        N_POST_TREATMENT=LARGE_N_POST, N_PRE_TREATMENT=LARGE_N_PRE, GLOBAL_SCALE=0.01
    )
    data = simulate_data(global_mean, seed, constants=constants)
    assert isinstance(data, Dataset)

    control_units = data.control_units
    treated_units = data.treated_units

    np.testing.assert_almost_equal(
        np.mean(control_units, axis=0), global_mean, decimal=0
    )
    np.testing.assert_almost_equal(
        np.mean(treated_units, axis=0), global_mean, decimal=0
    )


@given(
    n_control=st.integers(min_value=1, max_value=50),
    n_pre_treatment=st.integers(min_value=1, max_value=50),
    n_post_treatment=st.integers(min_value=1, max_value=50),
)
def test_array_shapes(n_control: int, n_pre_treatment: int, n_post_treatment: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post_treatment,
        N_PRE_TREATMENT=n_pre_treatment,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)

    # Test high-level property values
    assert data.n_units == n_control
    assert data.n_timepoints == n_pre_treatment + n_post_treatment
    assert data.n_pre_intervention == n_pre_treatment
    assert data.n_post_intervention == n_post_treatment

    # Test field shapes
    assert data.Xtr.shape == (n_pre_treatment, n_control)
    assert data.Xte.shape == (n_post_treatment, n_control)
    assert data.ytr.shape == (n_pre_treatment, 1)
    assert data.yte.shape == (n_post_treatment, 1)

    # Test property shapes
    Xtr, ytr = data.pre_intervention_obs
    Xte, yte = data.post_intervention_obs
    assert Xtr.shape == (n_pre_treatment, n_control)
    assert ytr.shape == (n_pre_treatment, 1)
    assert Xte.shape == (n_post_treatment, n_control)
    assert yte.shape == (n_post_treatment, 1)


@given(
    n_pre_treatment=st.integers(min_value=1, max_value=50),
    n_post_treatment=st.integers(min_value=1, max_value=50),
)
def test_indicator(n_pre_treatment: int, n_post_treatment: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post_treatment,
        N_PRE_TREATMENT=n_pre_treatment,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    assert data._get_indicator().sum() == n_post_treatment


@given(
    n_control=st.integers(min_value=1, max_value=50),
    n_pre_treatment=st.integers(min_value=1, max_value=50),
    n_post_treatment=st.integers(min_value=1, max_value=50),
)
def test_to_df(n_control: int, n_pre_treatment: int, n_post_treatment: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post_treatment,
        N_PRE_TREATMENT=n_pre_treatment,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)

    df = data.to_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (
        n_pre_treatment + n_post_treatment,
        n_control + NUM_NON_CONTROL_COLS,
    )

    colnames = data._get_columns()
    assert isinstance(colnames, list)
    assert colnames[0] == "T"
    assert len(colnames) == n_control + 1

    index = data.full_index
    assert isinstance(index, DatetimeIndex)
    assert index[0].strftime("%Y-%m-%d") == data._start_date.strftime("%Y-%m-%d")


@given(
    n_control=st.integers(min_value=2, max_value=50),
    n_pre_treatment=st.integers(min_value=10, max_value=50),
    n_post_treatment=st.integers(min_value=10, max_value=50),
    global_mean=st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
@settings(max_examples=5)
def test_to_azcausal(
    n_control: int, n_pre_treatment: int, n_post_treatment: int, global_mean: float
):
    constants = TestConstants(
        N_POST_TREATMENT=n_post_treatment,
        N_PRE_TREATMENT=n_pre_treatment,
        N_CONTROL=n_control,
    )
    data = simulate_data(global_mean, DEFAULT_SEED, constants=constants)

    panel = data.to_azcausal()
    model = DID()
    result = model.fit(panel)
    assert not np.isnan(result.effect.value)


@given(
    n_post_treatment=st.integers(min_value=10, max_value=50),
    n_pre_treatment=st.integers(min_value=10, max_value=50),
    idx=st.sampled_from(["pre-intervention", "post-intervention", "both"]),
)
def test_get_index(n_post_treatment: int, n_pre_treatment: int, idx: InterventionTypes):
    constants = TestConstants(
        N_POST_TREATMENT=n_post_treatment,
        N_PRE_TREATMENT=n_pre_treatment,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    idx_vals = data.get_index(idx)
    assert isinstance(idx_vals, DatetimeIndex)
    if idx == "both":
        assert len(idx_vals) == n_pre_treatment + n_post_treatment
    elif idx == "post-intervention":
        assert len(idx_vals) == n_post_treatment
    elif idx == "pre-intervention":
        assert len(idx_vals) == n_pre_treatment


@pytest.mark.parametrize("n_pre, n_post, n_control", [(60, 30, 10), (60, 30, 20)])
def test_drop_unit(n_pre: int, n_post: int, n_control: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post,
        N_PRE_TREATMENT=n_pre,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    desired_shape_Xtr = (n_pre, n_control - 1)
    desired_shape_Xte = (n_post, n_control - 1)
    desired_shape_ytr = (n_pre, 1)
    desired_shape_yte = (n_post, 1)

    for i in range(n_control):
        reduced_data = data.drop_unit(i)
        assert reduced_data.Xtr.shape == desired_shape_Xtr
        assert reduced_data.Xte.shape == desired_shape_Xte
        assert reduced_data.ytr.shape == desired_shape_ytr
        assert reduced_data.yte.shape == desired_shape_yte


@pytest.mark.parametrize("n_pre, n_post, n_control", [(60, 30, 10), (60, 30, 20)])
def test_to_placebo(n_pre: int, n_post: int, n_control: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post,
        N_PRE_TREATMENT=n_pre,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    desired_shape_Xtr = (n_pre, n_control - 1)
    desired_shape_Xte = (n_post, n_control - 1)
    desired_shape_ytr = (n_pre, 1)
    desired_shape_yte = (n_post, 1)

    for i in range(n_control):
        placebo_data = data.to_placebo_data(i)
        assert placebo_data.Xtr.shape == desired_shape_Xtr
        assert placebo_data.Xte.shape == desired_shape_Xte
        assert placebo_data.ytr.shape == desired_shape_ytr
        assert placebo_data.yte.shape == desired_shape_yte
        assert not data == placebo_data


@given(
    n_control=st.integers(min_value=2, max_value=50),
    n_pre_treatment=st.integers(min_value=10, max_value=50),
    n_post_treatment=st.integers(min_value=10, max_value=50),
    global_mean=st.floats(
        min_value=-5.0, max_value=5.0, allow_infinity=False, allow_nan=False
    ),
)
@settings(max_examples=10)
def test_eq(
    n_control: int, n_pre_treatment: int, n_post_treatment: int, global_mean: float
):
    constants = TestConstants(
        N_POST_TREATMENT=n_post_treatment,
        N_PRE_TREATMENT=n_pre_treatment,
        N_CONTROL=n_control,
    )
    data = simulate_data(global_mean, DEFAULT_SEED, constants=constants)
    copied_data = deepcopy(data)
    assert data == copied_data

    # Shape mismatch
    for i in range(n_control):
        reduced_data = data.drop_unit(i)
        assert not data == reduced_data


@pytest.mark.parametrize("n_pre, n_post, n_control", [(60, 30, 10), (60, 30, 20)])
def test_reassign_treatment(n_pre: int, n_post: int, n_control: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post,
        N_PRE_TREATMENT=n_pre,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    to_assign_ytr = np.ones(shape=(n_pre, 1))
    to_assign_yte = np.ones(shape=(n_post, 1))

    reassigned_data = reassign_treatment(data, to_assign_ytr, to_assign_yte)
    assert not data == reassigned_data
    np.testing.assert_equal(reassigned_data.ytr, to_assign_ytr)
    np.testing.assert_equal(reassigned_data.yte, to_assign_yte)


@given(
    name=st.text(
        alphabet=st.sampled_from(string.ascii_letters + string.digits + "_-"),
        min_size=1,
        max_size=10,
    ),
    extra_chars=st.text(
        alphabet=st.sampled_from(string.ascii_letters + string.digits + "_-"),
        min_size=1,
        max_size=10,
    ),
)
def test_naming_setter(name: str, extra_chars: str):
    data = simulate_data(10.0, DEFAULT_SEED)
    data.name = name
    assert data.name == name
    new_name = name + extra_chars
    data.name = new_name
    assert data.name == new_name


@given(
    seeds=st.lists(
        elements=st.integers(min_value=1, max_value=1000), min_size=1, max_size=10
    ),
    to_name=st.booleans(),
)
def test_dataset_container(seeds: tp.List[int], to_name: bool):
    datasets = [simulate_data(0.0, s) for s in seeds]
    if to_name:
        names = [f"D_{idx}" for idx in range(len(datasets))]
    else:
        names = None
    container = DatasetContainer(datasets, names)

    # Test names were correctly assigned
    if to_name:
        assert container.names == names
    else:
        assert container.names == [f"Dataset {idx}" for idx in range(len(datasets))]

    # Assert ordering
    for idx, dataset in enumerate(container):
        assert dataset == datasets[idx]

    # Assert no data was dropped/added
    assert len(container) == len(datasets)

    # Test `as_dict()` method preserves order
    container_dict = container.as_dict()
    for idx, (k, v) in enumerate(container_dict.items()):
        if to_name:
            assert k == names[idx]
        else:
            assert k == f"Dataset {idx}"
        assert v == datasets[idx]
