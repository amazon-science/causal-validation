from dataclasses import dataclass
import typing as tp

from azcausal.core.error import Error
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result

from causal_validation.data import Dataset


@dataclass
class AZCausalWrapper:
    model: Estimator
    error_estimator: tp.Optional[Error] = None

    def __post_init__(self):
        self._model_name = self.model.__class__.__name__

    def __call__(self, data: Dataset, **kwargs) -> Result:
        panel = data.to_azcausal()
        result = self.model.fit(panel, **kwargs)
        if self.error_estimator:
            self.model.error(result, self.error_estimator)
        return result
