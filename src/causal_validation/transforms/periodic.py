from dataclasses import dataclass
from typing import Tuple

from jaxtyping import Float
import numpy as np

from causal_validation.data import Dataset
from causal_validation.transforms.base import AdditiveTransform
from causal_validation.transforms.parameter import ParameterOrFloat


@dataclass(kw_only=True)
class Periodic(AdditiveTransform):
    amplitude: ParameterOrFloat = 1.0
    frequency: ParameterOrFloat = 1.0
    shift: ParameterOrFloat = 0.0
    offset: ParameterOrFloat = 0.0
    _slots: Tuple[str, str, str, str] = (
        "amplitude",
        "frequency",
        "shift",
        "offset",
    )

    def get_values(self, data: Dataset) -> Float[np.ndarray, "N D"]:
        amplitude = self.amplitude.get_value(**data._slots)
        frequency = self.frequency.get_value(**data._slots)
        shift = self.shift.get_value(**data._slots)
        offset = self.offset.get_value(**data._slots)
        x_vals = np.tile(
            np.linspace(0, 2 * np.pi, num=data.n_timepoints).reshape(-1, 1),
            reps=data.n_units + 1,
        )
        sine_curve = amplitude * np.sin((x_vals * np.abs(frequency)) + shift) + offset
        return sine_curve
