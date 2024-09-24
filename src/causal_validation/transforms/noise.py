from dataclasses import dataclass
from typing import Tuple

from jaxtyping import Float
import numpy as np
from scipy.stats import norm

from causal_validation.data import Dataset
from causal_validation.transforms.base import AdditiveTransform
from causal_validation.transforms.parameter import TimeVaryingParameter


@dataclass(kw_only=True)
class Noise(AdditiveTransform):
    """
    Transform the treatment by adding TimeVaryingParameter noise terms sampled from
    a specified sampling distribution. By default, the sampling distribution is
    Normal with 0 loc and 0.1 scale.
    """

    noise_dist: TimeVaryingParameter = TimeVaryingParameter(sampling_dist=norm(0, 0.1))
    _slots: Tuple[str] = ("noise_dist",)

    def get_values(self, data: Dataset) -> Float[np.ndarray, "N D"]:
        noise = np.zeros((data.n_timepoints, data.n_units + 1))
        noise_treatment = self.noise_dist.get_value(
            n_units=1, n_timepoints=data.n_timepoints
        ).reshape(-1)
        noise[:, 0] = noise_treatment
        return noise
