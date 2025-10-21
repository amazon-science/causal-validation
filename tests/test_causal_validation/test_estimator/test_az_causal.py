import typing as tp

from azcausal.core.effect import Effect
from azcausal.core.error import (
    Bootstrap,
    Error,
    JackKnife,
)
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result as _Result
from azcausal.estimators.panel import (
    did,
    sdid,
)
from hypothesis import (
    given,
    settings,
    strategies as st,
)
import numpy as np

from causal_validation.estimator import Result
from causal_validation.estimator.utils import AZCausalWrapper

from causal_validation.testing import (
    TestConstants,
    simulate_data,
)

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
    D = np.zeros((n_pre_treatment + n_post_treatment, n_control + 1))
    D[n_pre_treatment:, -1] = 1
    constants = TestConstants(
        TREATMENT_ASSIGNMENTS=D
    )
    data = simulate_data(global_mean=10.0, seed=seed, constants=constants)
    model = AZCausalWrapper(*model_error)
    result = model(data)

    assert isinstance(result, Result)
    assert isinstance(result.effect, Effect)
    assert not np.isnan(result.effect.value)
    assert isinstance(model._az_result, _Result)
    assert np.all(data.treated_unit_outputs == result.observed)
    assert (
        result.observed.shape == result.counterfactual.shape == result.synthetic.shape
    )