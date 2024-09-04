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
    assert len(result.effects) == n_control

    # Check the results are close to the true effect
    summary = result.summary()
    PlaceboSchema.validate(summary)
    assert isinstance(summary, pd.DataFrame)
    assert summary.shape == (1, 3)
    assert summary["Effect"].iloc[0] == pytest.approx(0.0, abs=0.1)
