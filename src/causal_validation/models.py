from dataclasses import dataclass
import typing as tp

from azcausal.core.effect import Effect
from azcausal.core.error import Error
from azcausal.core.estimator import Estimator

from causal_validation.data import Dataset


@dataclass
class AZCausalWrapper:
    model: Estimator
    error_estimator: tp.Optional[Error] = None

    def __call__(self, data: Dataset, **kwargs) -> Effect:
        panel = data.to_azcausal()
        result = self.model.fit(panel, **kwargs)
        if self.error_estimator:
            self.model.error(result, self.error_estimator)
        breakpoint()
        return result
