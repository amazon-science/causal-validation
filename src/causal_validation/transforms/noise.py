from dataclasses import (
    dataclass,
    field,
)
from typing import Tuple

from jaxtyping import Float
import numpy as np
from scipy.stats import norm

from causal_validation.data import Dataset
from causal_validation.transforms.base import (
    AdditiveCovariateTransform,
    AdditiveOutputTransform,
)
from causal_validation.transforms.parameter import (
    CovariateNoiseParameter,
    TimeAndUnitVaryingParameter,
)


@dataclass(kw_only=True)
class Noise(AdditiveOutputTransform):
    """
    Transform the treated units by adding TimeAndUnitVaryingParameter noise terms
    sampled from a specified sampling distribution. By default, the sampling
    distribution is Normal with 0 loc and 0.1 scale.
    """

    noise_dist: TimeAndUnitVaryingParameter = field(
        default_factory=lambda: TimeAndUnitVaryingParameter(sampling_dist=norm(0, 0.1))
    )
    _slots: Tuple[str] = ("noise_dist",)

    def get_values(self, data: Dataset) -> Float[np.ndarray, "T N"]:
        noise = np.zeros((data.n_timepoints, data.n_units))
        noise_treatment = self.noise_dist.get_value(
            n_units=data.n_treated_units, n_timepoints=data.n_timepoints
        )
        noise[:, data.treated_unit_indices] = noise_treatment
        return noise


@dataclass(kw_only=True)
class CovariateNoise(AdditiveCovariateTransform):
    """
    Transform the covariates by adding CovariateNoiseParameter noise terms sampled from
    a specified sampling distribution. By default, the sampling distribution is
    Normal with 0 loc and 0.1 scale.
    """

    noise_dist: CovariateNoiseParameter = field(
        default_factory=lambda: CovariateNoiseParameter(sampling_dist=norm(0, 0.1))
    )
    _slots: Tuple[str] = ("noise_dist",)

    def get_values(self, data: Dataset) -> Float[np.ndarray, "N D"]:
        noise = self.noise_dist.get_value(
            n_units=data.n_units,
            n_timepoints=data.n_timepoints,
            n_covariates=data.n_covariates,
        )
        return noise
