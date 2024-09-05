import numpy as np
import typing as tp
from causal_validation.models import AZCausalWrapper
from azcausal.estimators.panel import did, sdid
from azcausal.core.error import JackKnife, Bootstrap, Placebo
from hypothesis import given, settings, strategies as st
from azcausal.core.error import Error
from azcausal.core.estimator import Estimator
from causal_validation.testing import simulate_data, TestConstants
from azcausal.core.result import Result
from azcausal.core.effect import Effect

MODELS = [did.DID(), sdid.SDID()]
MODEL_ERROR = [
    (did.DID(), None),
    (sdid.SDID(), None),
    (sdid.SDID(), Bootstrap()),
    (sdid.SDID(), JackKnife()),
]


@given(
    model_error=st.sampled_from(MODEL_ERROR),
    n_control=st.integers(min_value=2, max_value=5),
    n_pre_treatment=st.integers(min_value=1, max_value=50),
    n_post_treatment=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=10)
def test_call(
    model_error: tp.Union[Estimator, Error],
    n_control: int,
    n_pre_treatment: int,
    n_post_treatment: int,
    seed: int,
):
    constancts = TestConstants(
        N_CONTROL=n_control,
        N_PRE_TREATMENT=n_pre_treatment,
        N_POST_TREATMENT=n_post_treatment,
    )
    data = simulate_data(global_mean=10.0, seed=seed, constants=constancts)
    model = AZCausalWrapper(*model_error)
    result = model(data)

    assert isinstance(result, Result)
    assert isinstance(result.effect, Effect)
    assert not np.isnan(result.effect.value)
