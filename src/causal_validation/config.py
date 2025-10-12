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
    TreatedSimulationTypes,
)


@dataclass(kw_only=True)
class Config:
    """Configuration for causal data generation.

    Args:
        treatment_assignments (Float[np.ndarray, "T N"]): Treatment assignments for T
            time steps and N units. Only supported with binary assignments.
        treated_simulation_type ("TreatedSimulationTypes"): Treated units can be
            simulated either "independent" of control units or "control-weighted",
            where waiting scheme is controlled by Dirichlet concentration parameter.
            Set to "control-weighted" by default.
        dirichlet_concentration (Number): Dirichlet parameters are set to a vector of
            dirichlet_concentration with length number of control units. This parameter
            controls how dense and sparse the generated weights are. Set to 1 by default
            and in effect only if treated_simulation_type is "control-weighted".
        n_covariates (Optional[int]): Number of covariates. Defaults to None.
        covariate_means (Optional[np.ndarray]): Normal dist. mean values for covariates.
            The lenght must be n_covariates. Defaults to None. If it is set to
            None while n_covariates is provided, covariate_means will be generated
            randomly from Normal distribution.
        covariate_stds (Optional[np.ndarray]): Normal dist. std values for covariates.
            The lenght must be n_covariates. Defaults to None. If it is set to
            None while n_covariates is provided, covariate_stds will be generated
            randomly from Half-Cauchy distribution.
        covariate_coeffs (Optional[np.ndarray]): Linear regression
            coefficients to map covariates to output observations. K is n_covariates.
            Defaults to None.
        global_mean (Number): Global mean for data generation. Defaults to 20.0.
        global_scale (Number): Global scale for data generation. Defaults to 0.2.
        start_date (dt.date): Start date for time series. Defaults to 2023-01-01.
        seed (int): Random seed for reproducibility. Defaults to 123.
        weights (Optional[list[np.ndarray]]): Length num of treateds list of weights.
            Each element is length num of control, indicating how to weigh control
            units to generate treated.
    """

    treatment_assignments: Float[np.ndarray, "T N"]
    treated_simulation_type: "TreatedSimulationTypes" = "control-weighted"
    dirichlet_concentration: Number = 1.0
    n_covariates: tp.Optional[int] = None
    covariate_means: tp.Optional[np.ndarray] = None
    covariate_stds: tp.Optional[np.ndarray] = None
    covariate_coeffs: tp.Optional[np.ndarray] = None
    global_mean: Number = 20.0
    global_scale: Number = 0.2
    start_date: dt.date = dt.date(year=2023, month=1, day=1)
    seed: int = 123
    weights: tp.Optional[list[np.ndarray]] = None

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
        if self.covariate_means is not None:
            assert self.covariate_means.shape == (
                self.n_covariates,
            )

        if self.covariate_stds is not None:
            assert self.covariate_stds.shape == (
                self.n_covariates,
            )

        if (self.n_covariates is not None) & (self.covariate_means is None):
            self.covariate_means = self.rng.normal(
                loc=0.0, scale=5.0, size=(self.n_covariates)
            )

        if (self.n_covariates is not None) & (self.covariate_stds is None):
            self.covariate_stds = halfcauchy.rvs(
                scale=0.5,
                size=(self.n_covariates),
                random_state=self.rng,
            )

        if (self.n_covariates is not None) & (self.covariate_coeffs is None):
            self.covariate_coeffs = self.rng.normal(
                loc=0.0, scale=5.0, size=self.n_covariates
            )

        n_units = self.treatment_assignments.shape[1]
        treated_units = [i for i in range(n_units) if any(self.treatment_assignments[:, i] != 0)]
        n_treated_units = len(treated_units)
        n_control_units = n_units - n_treated_units

        if self.treated_simulation_type == "control-weighted":
            if self.weights is None:
                self.weights = [
                    self.rng.dirichlet(
                        self.dirichlet_concentration * np.ones(n_control_units)
                    )
                    for _ in range(n_treated_units)
                ]
            else:
                assert len(self.weights) == n_treated_units
                assert all(
                    [
                        len(w) == n_control_units
                        for w in self.weights
                    ]
                )
