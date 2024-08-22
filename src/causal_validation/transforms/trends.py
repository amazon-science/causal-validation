from dataclasses import dataclass
from typing import Tuple

from jaxtyping import Float
import numpy as np

from causal_validation.data import Dataset
from causal_validation.transforms.base import AdditiveTransform
from causal_validation.transforms.parameter import ParameterOrFloat


@dataclass(kw_only=True)
class Trend(AdditiveTransform):
    degree: int = 1
    coefficient: ParameterOrFloat = 1.0
    intercept: ParameterOrFloat = 0.0
    _slots: Tuple[str, str] = ("coefficient", "intercept")

    def get_values(self, data: Dataset) -> Float[np.ndarray, "N D"]:
        coefficient = self._resolve_parameter(data, self.coefficient)
        intercept = self._resolve_parameter(data, self.intercept)
        trend = np.tile(
            np.arange(data.n_timepoints)[:, None] ** self.degree, data.n_units + 1
        )
        scaled_trend = intercept + coefficient * trend
        return scaled_trend
