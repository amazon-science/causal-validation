from copy import deepcopy
import datetime as dt
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
)
from causal_validation.types import InterventionTypes


@given(
    T=st.integers(min_value=10, max_value=5000),
    N=st.integers(min_value=5, max_value=5000),
    K=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=10)
def test_dataset(T: int, N: int, K: int, seed: int):
    D = np.zeros((T, N))
    D[6:, 2] = 1
    D[8:, 3] = 1

    rng = np.random.RandomState(seed)
    X = rng.randn(T, N, K)
    Y = rng.randn(T, N)

    data1 = Dataset(Y, D, X)
    data2 = Dataset(Y, D, X)
    data3 = Dataset(Y, D)

    assert data1 == data2
    assert data1 != data3

    assert data1.n_units == N
    assert data1.n_timepoints == T
    assert data1.n_covariates == K

    assert data1.n_post_intervention[0] == 0
    assert data1.n_post_intervention[-1] == 0
    assert data1.n_post_intervention[2] == T - 6
    assert data1.n_post_intervention[3] == T - 8

    assert (
        data1.n_pre_intervention
        == (T * np.ones(N) - data1.n_post_intervention).tolist()
    )

    assert data1.n_treated_units == 2
    assert data1.n_control_units == N - 2
    assert data1.treated_unit_indices == [2, 3]
    assert data1.control_unit_indices == [0, 1] + list(range(4, N))

    assert np.all(data1.treated_unit_outputs == Y[:, 2:4])
    assert np.all(
        data1.control_unit_outputs == np.concatenate([Y[:, :2], Y[:, 4:]], axis=1)
    )

    assert np.all(data1.treated_unit_covariates == X[:, 2:4, :])
    assert np.all(
        data1.control_unit_covariates
        == np.concatenate([X[:, :2, :], X[:, 4:, :]], axis=1)
    )


@given(
    T=st.integers(min_value=2, max_value=5000),
    N=st.integers(min_value=2, max_value=5000),
    K=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=10)
def test_dataset_to_df(T: int, N: int, K: int, seed: int):
    rng = np.random.RandomState(seed)
    X = rng.randn(T, N, K)
    Y = rng.randn(T, N)
    D = rng.randn(T, N)

    data1 = Dataset(Y, D, X)
    df1 = data1.to_df()
    assert isinstance(df1, pd.DataFrame)
    assert df1.shape == (T, N * (K + 2))
    assert df1.columns.nlevels == 2
    assert df1.columns[0] == ("U0", "Y")
    assert df1.columns[1] == ("U0", "D")
    assert df1.columns[2] == ("U0", "X0")
    assert df1.columns[K + 2] == ("U1", "Y")
    assert df1.columns[K + 3] == ("U1", "D")
    assert df1.columns[K + 4] == ("U1", "X0")

    C1 = df1.to_numpy()
    C1[:, 0] = Y[:, 0]
    C1[:, 1] = D[:, 0]
    C1[:, 2] = X[:, 0, 0]
    C1[:, K + 2] = Y[:, 1]
    C1[:, K + 3] = D[:, 1]
    C1[:, K + 4] = X[:, 1, 0]
    C1[:, -1] = X[:, -1, -1]
    C1[:, -K - 2] = Y[:, -1]

    data2 = Dataset(Y, D)
    df2 = data2.to_df()
    assert isinstance(df2, pd.DataFrame)
    assert df2.shape == (T, N * 2)
    assert df2.columns.nlevels == 2
    assert df2.columns[0] == ("U0", "Y")
    assert df2.columns[1] == ("U0", "D")
    assert df2.columns[2] == ("U1", "Y")
    assert df2.columns[3] == ("U1", "D")


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
    T=st.integers(min_value=2, max_value=5000),
    N=st.integers(min_value=2, max_value=5000),
)
def test_naming_setter(name: str, extra_chars: str, T: int, N: int):
    Y = np.random.randn(T, N)
    D = np.random.randn(T, N)
    data = Dataset(Y, D)
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
    T=st.integers(min_value=2, max_value=5000),
    N=st.integers(min_value=2, max_value=5000),
)
def test_dataset_container(seeds: tp.List[int], to_name: bool, T: int, N: int):
    datasets = []
    for s in seeds:
        rng = np.random.RandomState(s)
        Y = rng.randn(T, N)
        D = rng.randn(T, N)
        data = Dataset(Y, D)
        datasets.append(data)

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


@given(
    T=st.integers(min_value=2, max_value=100),
    N=st.integers(min_value=2, max_value=100),
    K=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=1, max_value=100),
    use_bernoulli=st.booleans(),
    include_X=st.booleans(),
)
@settings(max_examples=10)
def test_inflate(
    T: int, N: int, K: int, seed: int, use_bernoulli: bool, include_X: bool
):
    rng = np.random.RandomState(seed)
    Y = rng.randn(T, N)
    D = rng.binomial(1, 0.3, (T, N)) if use_bernoulli else np.abs(rng.randn(T, N))
    X = rng.randn(T, N, K) if include_X else None

    data = Dataset(Y, D, X)
    inflation_vals = rng.uniform(0.01, 1.0, (T, N))

    inflated_data = data.inflate(inflation_vals)

    assert np.allclose(inflated_data.Y, Y * inflation_vals)
    assert np.allclose(inflated_data.D, D)
    if include_X:
        assert np.allclose(inflated_data.X, X)
    else:
        assert inflated_data.X is None
