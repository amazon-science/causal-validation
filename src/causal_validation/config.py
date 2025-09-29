from dataclasses import (
    dataclass,
    field,
)
import datetime as dt
import typing as tp

from jaxtyping import Float
import numpy as np
from scipy.stats import halfcauchy

from causal_validation.types import (
    Number,
    WeightTypes,
)
from causal_validation.weights import UniformWeights


@dataclass(kw_only=True, frozen=True)
class WeightConfig:
    weight_type: "WeightTypes" = field(default_factory=UniformWeights)


@dataclass(kw_only=True)
class Config:
    """Configuration for causal data generation.

    Args:
        n_control_units (int): Number of control units in the synthetic dataset.
        n_pre_intervention_timepoints (int): Number of time points before intervention.
        n_post_intervention_timepoints (int): Number of time points after intervention.
        n_covariates (Optional[int]): Number of covariates. Defaults to None.
        covariate_means (Optional[Float[np.ndarray, "D K"]]): Mean values for covariates
            D is n_control_units and K is n_covariates. Defaults to None. If it is set
            to None while n_covariates is provided, covariate_means will be generated
            randomly from Normal distribution.
        covariate_stds (Optional[Float[np.ndarray, "D K"]]): Standard deviations for
            covariates. D is n_control_units and K is n_covariates. Defaults to None.
            If it is set to None while n_covariates is provided, covariate_stds
            will be generated randomly from Half-Cauchy distribution.
        covariate_coeffs (Optional[np.ndarray]): Linear regression
            coefficients to map covariates to output observations. K is n_covariates.
            Defaults to None.
        global_mean (Number): Global mean for data generation. Defaults to 20.0.
        global_scale (Number): Global scale for data generation. Defaults to 0.2.
        start_date (dt.date): Start date for time series. Defaults to 2023-01-01.
        seed (int): Random seed for reproducibility. Defaults to 123.
        weights_cfg (WeightConfig): Configuration for unit weights. Defaults to
            UniformWeights.
    """

    n_control_units: int
    n_pre_intervention_timepoints: int
    n_post_intervention_timepoints: int
    n_covariates: tp.Optional[int] = None
    covariate_means: tp.Optional[Float[np.ndarray, "D K"]] = None
    covariate_stds: tp.Optional[Float[np.ndarray, "D K"]] = None
    covariate_coeffs: tp.Optional[np.ndarray] = None
    global_mean: Number = 20.0
    global_scale: Number = 0.2
    start_date: dt.date = dt.date(year=2023, month=1, day=1)
    seed: int = 123
    weights_cfg: WeightConfig = field(default_factory=WeightConfig)

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
        if self.covariate_means is not None:
            assert self.covariate_means.shape == (
                self.n_control_units,
                self.n_covariates,
            )

        if self.covariate_stds is not None:
            assert self.covariate_stds.shape == (
                self.n_control_units,
                self.n_covariates,
            )

        if (self.n_covariates is not None) & (self.covariate_means is None):
            self.covariate_means = self.rng.normal(
                loc=0.0, scale=5.0, size=(self.n_control_units, self.n_covariates)
            )

        if (self.n_covariates is not None) & (self.covariate_stds is None):
            self.covariate_stds = halfcauchy.rvs(
                scale=0.5,
                size=(self.n_control_units, self.n_covariates),
                random_state=self.rng,
            )

        if (self.n_covariates is not None) & (self.covariate_coeffs is None):
            self.covariate_coeffs = self.rng.normal(
                loc=0.0, scale=5.0, size=self.n_covariates
            )
