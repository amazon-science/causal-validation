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

from causal_validation.effects import StaticEffect
from causal_validation.estimator.utils import AZCausalWrapper
from causal_validation.testing import (
    TestConstants,
    simulate_data,
)
from causal_validation.transforms import Trend
from causal_validation.validation.rmspe import (
    RMSPESchema,
    RMSPETest,
    RMSPETestResult,
)
from causal_validation.validation.testing import (
    RMSPETestStatistic,
    TestResult,
    TestResultFrame,
)

N_TIME_POINTS = 100
N_PRE_TREATMENT = 50


def test_schema_coerce():
    df = RMSPESchema.example()
    cols = df.columns
    for col in cols:
        if col not in ["Model", "Dataset"]:
            df[col] = np.ceil((df[col]))
            RMSPESchema.validate(df)


@given(
    global_mean=st.floats(min_value=0.0, max_value=10.0),
    seed=st.integers(min_value=0, max_value=1000000),
    n_control=st.integers(min_value=10, max_value=20),
    cf_inflate=st.one_of(
        st.floats(min_value=1e-10, max_value=2.0),
        st.floats(min_value=-2.0, max_value=-1e-10),
    ),
    s_inflate=st.one_of(
        st.floats(min_value=1e-10, max_value=2.0),
        st.floats(min_value=-2.0, max_value=-1e-10),
    ),
)
@settings(max_examples=10)
def test_rmspe_test_stat(
    global_mean: float, seed: int, n_control: int, cf_inflate: float, s_inflate: float
):
    # Simulate data
    D = np.zeros((1000, n_control + 1))
    D[N_PRE_TREATMENT:, -1] = 1
    constants = TestConstants(
        TREATMENT_ASSIGNMENTS=D,
        DIRICHLET_CONCENTRATION=10000,
        N_COVARIATES=0,
        GLOBAL_SCALE=0.001,
    )
    data = simulate_data(global_mean=global_mean, seed=seed, constants=constants)
    rmspe = RMSPETestStatistic()
    counterfactual = data.treated_unit_outputs + cf_inflate
    synthetic = counterfactual
    assert rmspe(data, counterfactual, synthetic, N_PRE_TREATMENT) == pytest.approx(1.0)

    synthetic = data.treated_unit_outputs + s_inflate
    assert rmspe(data, counterfactual, synthetic, N_PRE_TREATMENT) == pytest.approx(
        abs(cf_inflate) / abs(s_inflate)
    )

    synthetic = data.treated_unit_outputs
    with pytest.raises(
        ZeroDivisionError, match="Error: pre intervention period MSPE is 0!"
    ):
        rmspe(data, counterfactual, synthetic, N_PRE_TREATMENT)


@given(
    global_mean=st.floats(min_value=0.0, max_value=10.0),
    effect=st.one_of(
        st.floats(min_value=1.0, max_value=5.0),
        st.floats(min_value=-5.0, max_value=-1.0),
    ),
    seed=st.integers(min_value=0, max_value=1000000),
    n_control=st.integers(min_value=10, max_value=20),
    model=st.sampled_from([DID(), SDID()]),
)
@settings(max_examples=10)
def test_rmspe_test(
    global_mean: float,
    effect: float,
    seed: int,
    n_control: int,
    model: tp.Union[DID, SDID],
):
    # Simulate data with a trend and effect
    D = np.zeros((N_TIME_POINTS, n_control + 1))
    D[N_PRE_TREATMENT:, -1] = 1
    constants = TestConstants(
        TREATMENT_ASSIGNMENTS=D,
        DIRICHLET_CONCENTRATION=10000,
        N_COVARIATES=0,
        GLOBAL_SCALE=0.001,
    )
    data = simulate_data(global_mean=global_mean, seed=seed, constants=constants)
    trend_term = Trend(degree=1, coefficient=0.1)
    static_effect = StaticEffect(effect=effect)
    data = static_effect(trend_term(data))

    model = AZCausalWrapper(model)
    result = RMSPETest(model, data).execute()

    assert isinstance(result, RMSPETestResult)
    assert isinstance(result, TestResultFrame)
    assert set(result.treatment_test_results.keys()) == set(
        result.pseudo_treatment_test_statistics.keys()
    )

    for k, v in result.treatment_test_results.items():
        assert isinstance(v, TestResult)
        assert len(result.pseudo_treatment_test_statistics[k]) == n_control

    summary = result.to_df()
    RMSPESchema.validate(summary)
    assert isinstance(summary, pd.DataFrame)
    assert summary.shape == (1, 4)
    assert summary["p-value"].iloc[0] == pytest.approx(1.0 / (n_control + 1))

    rich_summary = result.summary()
    assert isinstance(rich_summary, Table)
    n_rows = result.summary().row_count
    assert n_rows == summary.shape[0]


@pytest.mark.parametrize("n_control", [9, 10])
def test_multiple_models(n_control: int):
    D = np.zeros((N_TIME_POINTS, n_control + 1))
    D[N_PRE_TREATMENT:, -1] = 1
    constants = TestConstants(
        TREATMENT_ASSIGNMENTS=D,
        DIRICHLET_CONCENTRATION=10000,
        N_COVARIATES=0,
        GLOBAL_SCALE=0.001,
    )
    data = simulate_data(global_mean=20.0, seed=123, constants=constants)
    trend_term = Trend(degree=1, coefficient=0.1)
    data = trend_term(data)

    model1 = AZCausalWrapper(DID())
    model2 = AZCausalWrapper(SDID())
    result = RMSPETest([model1, model2], data).execute()

    result_df = result.to_df()
    result_rich = result.summary()
    assert result_df.shape == (2, 4)
    assert result_df.shape[0] == result_rich.row_count
    assert result_df["Model"].tolist() == ["DID", "SDID"]
    for k, v in result.treatment_test_results.items():
        assert isinstance(v, TestResult)
        assert len(result.pseudo_treatment_test_statistics[k]) == n_control


@given(
    seeds=st.lists(
        elements=st.integers(min_value=1, max_value=1000), min_size=1, max_size=5
    )
)
@settings(max_examples=5)
def test_multiple_datasets(seeds: tp.List[int]):
    D = np.zeros((N_TIME_POINTS, 40))
    D[N_PRE_TREATMENT:, -1] = 1
    constants = TestConstants(TREATMENT_ASSIGNMENTS=D)
    data = [simulate_data(global_mean=20.0, seed=s, constants=constants) for s in seeds]
    n_data = len(data)

    model = AZCausalWrapper(DID())
    result = RMSPETest(model, data).execute()

    result_df = result.to_df()
    result_rich = result.summary()
    assert result_df.shape == (n_data, 4)
    assert result_df.shape[0] == result_rich.row_count
