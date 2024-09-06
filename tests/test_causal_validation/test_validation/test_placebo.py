import typing as tp

from azcausal.estimators.panel.did import DID
from azcausal.estimators.panel.sdid import SDID
from hypothesis import (
    given,
    settings,
    strategies as st,
)
import numpy as np
import pandas as pd
import pytest
from rich.table import Table

from causal_validation.models import AZCausalWrapper
from causal_validation.testing import (
    TestConstants,
    simulate_data,
)
from causal_validation.transforms import Trend
from causal_validation.validation.placebo import (
    PlaceboSchema,
    PlaceboTest,
    PlaceboTestResult,
)


def test_schema_coerce():
    df = PlaceboSchema.example()
    cols = df.columns
    for col in cols:
        if col != "Model":
            df[col] = np.ceil((df[col]))
            PlaceboSchema.validate(df)


@given(
    global_mean=st.floats(min_value=0.0, max_value=10.0),
    seed=st.integers(min_value=0, max_value=1000000),
    n_control=st.integers(min_value=10, max_value=20),
    model=st.sampled_from([DID(), SDID()]),
)
@settings(max_examples=10)
def test_placebo_test(
    global_mean: float, seed: int, n_control: int, model: tp.Union[DID, SDID]
):
    # Simulate data with a trend
    constants = TestConstants(N_CONTROL=n_control, GLOBAL_SCALE=0.001)
    data = simulate_data(global_mean=global_mean, seed=seed, constants=constants)
    trend_term = Trend(degree=1, coefficient=0.1)
    data = trend_term(data)

    # Execute the placebo test
    model = AZCausalWrapper(model)
    result = PlaceboTest(model, data).execute()

    # Check that the structure of result
    assert isinstance(result, PlaceboTestResult)
    for _, v in result.effects.items():
        assert len(v) == n_control

    # Check the results are close to the true effect
    summary = result.to_df()
    PlaceboSchema.validate(summary)
    assert isinstance(summary, pd.DataFrame)
    assert summary.shape == (1, 5)
    assert summary["Effect"].iloc[0] == pytest.approx(0.0, abs=0.1)

    rich_summary = result.summary()
    assert isinstance(rich_summary, Table)
    n_rows = result.summary().row_count
    assert n_rows == summary.shape[0]


@pytest.mark.parametrize("n_control", [9, 10])
def test_multiple_models(n_control: int):
    constants = TestConstants(N_CONTROL=n_control, GLOBAL_SCALE=0.001)
    data = simulate_data(global_mean=20.0, seed=123, constants=constants)
    trend_term = Trend(degree=1, coefficient=0.1)
    data = trend_term(data)

    model1 = AZCausalWrapper(DID())
    model2 = AZCausalWrapper(SDID())
    result = PlaceboTest([model1, model2], data).execute()

    result_df = result.to_df()
    result_rich = result.summary()
    assert result_df.shape == (2, 5)
    assert result_df.shape[0] == result_rich.row_count
    assert result_df["Model"].tolist() == ["DID", "SDID"]
    for _, v in result.effects.items():
        assert len(v) == n_control
