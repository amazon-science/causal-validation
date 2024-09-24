from dataclasses import dataclass
import typing as tp

from azcausal.core.effect import Effect
from azcausal.core.error import Error
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result as _Result
from jaxtyping import Float

from causal_validation.data import Dataset
from causal_validation.types import NPArray


@dataclass
class Result:
    effect: Effect
    counterfactual: Float[NPArray, "N 1"]
    synthetic: Float[NPArray, "N 1"]
    observed: Float[NPArray, "N 1"]


@dataclass
class AZCausalWrapper:
    model: Estimator
    error_estimator: tp.Optional[Error] = None
    _az_result: _Result = None

    def __post_init__(self):
        self._model_name = self.model.__class__.__name__

    def __call__(self, data: Dataset, **kwargs) -> Result:
        panel = data.to_azcausal()
        result = self.model.fit(panel, **kwargs)
        if self.error_estimator:
            self.model.error(result, self.error_estimator)
        self._az_result = result

        res = Result(
            effect=result.effect,
            counterfactual=self.counterfactual,
            synthetic=self.synthetic,
            observed=self.observed,
        )
        return res

    @property
    def counterfactual(self) -> Float[NPArray, "N 1"]:
        df = self._az_result.effect.by_time
        c_factual = df.loc[:, "CF"].values.reshape(-1, 1)
        return c_factual

    @property
    def synthetic(self) -> Float[NPArray, "N 1"]:
        df = self._az_result.effect.by_time
        synth_control = df.loc[:, "C"].values.reshape(-1, 1)
        return synth_control

    @property
    def observed(self) -> Float[NPArray, "N 1"]:
        df = self._az_result.effect.by_time
        treated = df.loc[:, "T"].values.reshape(-1, 1)
        return treated
