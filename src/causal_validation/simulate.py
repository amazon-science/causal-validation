import typing as tp

import numpy as np

from causal_validation.config import Config
from causal_validation.data import Dataset


def simulate(config: Config, key: tp.Optional[np.random.RandomState] = None) -> Dataset:
    if key is None:
        key = config.rng

    if config.treated_simulation_type == "independent":
        base_data = _simulate_with_independent_treated_units(config, key)
    elif config.treated_simulation_type == "control-weighted":
        base_data = _simulate_with_control_weighted_treated_units(config, key)
    else:
        raise ValueError(
            f"Unknown treated simulation type: {config.treated_simulation_type}"
        )
    return base_data


def _simulate_with_independent_treated_units(
    config: Config, key: np.random.RandomState
) -> Dataset:
    n_timepoints, n_units = config.treatment_assignments.shape

    Y = key.normal(
        loc=config.global_mean, scale=config.global_scale, size=(n_timepoints, n_units)
    )

    if config.n_covariates is not None:
        X = key.normal(
            loc=config.covariate_means,
            scale=config.covariate_stds,
            size=(n_timepoints, n_units, config.n_covariates),
        )

        Y = Y + X @ config.covariate_coeffs
    else:
        X = None

    data = Dataset(
        Y,
        config.treatment_assignments,
        X,
        _start_date=config.start_date,
    )

    return data


def _simulate_with_control_weighted_treated_units(
    config: Config, key: np.random.RandomState
) -> Dataset:
    n_timepoints, n_units = config.treatment_assignments.shape

    Y = np.zeros((n_timepoints, n_units))
    if config.n_covariates is not None:
        X = np.zeros((n_timepoints, n_units, config.n_covariates))
    else:
        X = None
    data_void = Dataset(
        Y,
        config.treatment_assignments,
        X,
        _start_date=config.start_date,
    )

    n_control_units = data_void.n_control_units
    control_unit_indices = data_void.control_unit_indices
    treated_unit_indices = data_void.treated_unit_indices

    Y_control = key.normal(
        loc=config.global_mean,
        scale=config.global_scale,
        size=(n_timepoints, n_control_units),
    )

    if config.n_covariates is not None:
        X_control = key.normal(
            loc=config.covariate_means,
            scale=config.covariate_stds,
            size=(n_timepoints, n_control_units, config.n_covariates),
        )
        Y_control = Y_control + X_control @ config.covariate_coeffs
        X[:, control_unit_indices, :] = X_control
        for i, weights in enumerate(config.weights):
            X_treated_i = np.einsum("ijk,j->ik", X_control, weights)
            X[:, treated_unit_indices[i], :] = X_treated_i

    Y[:, control_unit_indices] = Y_control
    for i, weights in enumerate(config.weights):
        Y_treated_i = Y_control @ weights
        Y[:, treated_unit_indices[i]] = Y_treated_i

    data = Dataset(
        Y,
        config.treatment_assignments,
        X,
        _start_date=config.start_date,
    )

    return data
