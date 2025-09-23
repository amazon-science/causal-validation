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
import datetime as dt

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
        
        assert reduced_data.counterfactual == data.counterfactual
        assert reduced_data.synthetic == data.synthetic
        assert reduced_data._name == data._name


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
    n_pre=st.integers(min_value=10, max_value=100),
    n_post=st.integers(min_value=10, max_value=100),
    n_control=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=5)
def test_counterfactual_synthetic_attributes(n_pre: int, n_post: int, n_control: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post,
        N_PRE_TREATMENT=n_pre,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    
    assert data.counterfactual is None
    assert data.synthetic is None
    
    counterfactual_vals = np.random.randn(n_post, 1)
    synthetic_vals = np.random.randn(n_post, 1)
    
    data_with_attrs = Dataset(
        data.Xtr, data.Xte, data.ytr, data.yte, data._start_date,
        data.Ptr, data.Pte, data.Rtr, data.Rte,
        counterfactual_vals, synthetic_vals, "test_dataset"
    )
    
    np.testing.assert_array_equal(data_with_attrs.counterfactual, counterfactual_vals)
    np.testing.assert_array_equal(data_with_attrs.synthetic, synthetic_vals)
    assert data_with_attrs.name == "test_dataset"


@given(
    n_pre=st.integers(min_value=10, max_value=100),
    n_post=st.integers(min_value=10, max_value=100),
    n_control=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=5)
def test_inflate_method(n_pre: int, n_post: int, n_control: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post,
        N_PRE_TREATMENT=n_pre,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    
    inflation_vals = np.ones((n_post, 1)) * 1.1 
    inflated_data = data.inflate(inflation_vals)
    
    np.testing.assert_array_equal(inflated_data.Xtr, data.Xtr)
    np.testing.assert_array_equal(inflated_data.ytr, data.ytr)
    np.testing.assert_array_equal(inflated_data.Xte, data.Xte)
    
    expected_yte = data.yte * inflation_vals
    np.testing.assert_array_equal(inflated_data.yte, expected_yte)
    
    np.testing.assert_array_equal(inflated_data.counterfactual, data.yte)


@given(
    n_pre=st.integers(min_value=10, max_value=100),
    n_post=st.integers(min_value=10, max_value=100),
    n_control=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=5)
def test_control_treated_properties(n_pre: int, n_post: int, n_control: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post,
        N_PRE_TREATMENT=n_pre,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    
    control_units = data.control_units
    expected_control = np.vstack([data.Xtr, data.Xte])
    np.testing.assert_array_equal(control_units, expected_control)
    assert control_units.shape == (n_pre + n_post, n_control)
    
    treated_units = data.treated_units
    expected_treated = np.vstack([data.ytr, data.yte])
    np.testing.assert_array_equal(treated_units, expected_treated)
    assert treated_units.shape == (n_pre + n_post, 1)


@given(
    n_pre=st.integers(min_value=10, max_value=100),
    n_post=st.integers(min_value=10, max_value=100),
    n_control=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=5)
def test_covariate_properties_without_covariates(n_pre: int, n_post: int, n_control: int):
    constants = TestConstants(
        N_POST_TREATMENT=n_post,
        N_PRE_TREATMENT=n_pre,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    
    assert data.has_covariates is False
    assert data.control_covariates is None
    assert data.treated_covariates is None
    assert data.pre_intervention_covariates is None
    assert data.post_intervention_covariates is None
    assert data.n_covariates == 0


@given(
    n_pre=st.integers(min_value=10, max_value=50),
    n_post=st.integers(min_value=10, max_value=50),
    n_control=st.integers(min_value=2, max_value=10),
    n_covariates=st.integers(min_value=1, max_value=5),
    Xtr=st.data(),
    Xte=st.data(),
    ytr=st.data(),
    yte=st.data(),
    Ptr=st.data(),
    Pte=st.data(),
    Rtr=st.data(),
    Rte=st.data(),
)
@settings(max_examples=5)
def test_covariate_properties_with_covariates(n_pre: int, 
                                              n_post: int, 
                                              n_control: int, 
                                              n_covariates: int, 
                                              Xtr, 
                                              Xte, 
                                              ytr, 
                                              yte, 
                                              Ptr, 
                                              Pte, 
                                              Rtr, 
                                              Rte):
    
    Xtr = Xtr.draw(st.lists(st.floats(min_value=-10, max_value=10), 
                            min_size=n_pre*n_control, max_size=n_pre*n_control))
    Xtr = np.array(Xtr).reshape(n_pre, n_control)
    
    Xte = Xte.draw(st.lists(st.floats(min_value=-10, max_value=10), 
                            min_size=n_post*n_control, max_size=n_post*n_control))
    Xte = np.array(Xte).reshape(n_post, n_control)
    
    ytr = ytr.draw(st.lists(st.floats(min_value=-10, max_value=10), 
                            min_size=n_pre, max_size=n_pre))
    ytr = np.array(ytr).reshape(n_pre, 1)
    
    yte = yte.draw(st.lists(st.floats(min_value=-10, max_value=10), 
                            min_size=n_post, max_size=n_post))
    yte = np.array(yte).reshape(n_post, 1)
    
    Ptr = Ptr.draw(st.lists(st.floats(min_value=-10, max_value=10), 
                            min_size=n_pre*n_control*n_covariates, max_size=n_pre*n_control*n_covariates))
    Ptr = np.array(Ptr).reshape(n_pre, n_control, n_covariates)
    
    Pte = Pte.draw(st.lists(st.floats(min_value=-10, max_value=10), 
                            min_size=n_post*n_control*n_covariates, max_size=n_post*n_control*n_covariates))
    Pte = np.array(Pte).reshape(n_post, n_control, n_covariates)
    
    Rtr = Rtr.draw(st.lists(st.floats(min_value=-10, max_value=10), 
                            min_size=n_pre*n_covariates, max_size=n_pre*n_covariates))
    Rtr = np.array(Rtr).reshape(n_pre, 1, n_covariates)
    
    Rte = Rte.draw(st.lists(st.floats(min_value=-10, max_value=10), 
                            min_size=n_post*n_covariates, max_size=n_post*n_covariates))
    Rte = np.array(Rte).reshape(n_post, 1, n_covariates)
    
    data = Dataset(Xtr, Xte, ytr, yte, dt.date(2023, 1, 1), Ptr, Pte, Rtr, Rte)
    
    assert data.n_covariates == n_covariates
    assert data.has_covariates is True
    
    control_covariates = data.control_covariates
    expected_control_cov = np.vstack([Ptr, Pte])
    np.testing.assert_array_equal(control_covariates, expected_control_cov)
    assert control_covariates.shape == (n_pre + n_post, n_control, n_covariates)
    
    treated_covariates = data.treated_covariates
    expected_treated_cov = np.vstack([Rtr, Rte])
    np.testing.assert_array_equal(treated_covariates, expected_treated_cov)
    assert treated_covariates.shape == (n_pre + n_post, 1, n_covariates)
    
    pre_cov = data.pre_intervention_covariates
    assert pre_cov == (Ptr, Rtr)
    
    post_cov = data.post_intervention_covariates
    assert post_cov == (Pte, Rte)

