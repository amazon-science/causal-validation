import typing as tp

import numpy as np

from causal_validation.config import Config
from causal_validation.data import Dataset
from causal_validation.weights import (
    AbstractWeights,
    UniformWeights,
)


def simulate(config: Config, key: tp.Optional[np.random.RandomState] = None) -> Dataset:
    if key is None:
        key = config.rng
    weights = UniformWeights()

    base_data = _simulate_base_obs(config, weights, key)
    return base_data


def _simulate_base_obs(
    config: Config, weights: AbstractWeights, key: np.random.RandomState
) -> Dataset:
    n_timepoints = (
        config.n_pre_intervention_timepoints + config.n_post_intervention_timepoints
    )
    n_units = config.n_control_units
    obs = key.normal(
        loc=config.global_mean, scale=config.global_scale, size=(n_timepoints, n_units)
    )

    if config.n_covariates is not None:
        Xtr_ = obs[: config.n_pre_intervention_timepoints, :]
        Xte_ = obs[config.n_pre_intervention_timepoints :, :]

        covariates = key.normal(
            loc=config.covariate_means,
            scale=config.covariate_stds,
            size=(n_timepoints, n_units, config.n_covariates),
        )

        Ptr = covariates[: config.n_pre_intervention_timepoints, :, :]
        Pte = covariates[config.n_pre_intervention_timepoints :, :, :]

        Xtr = Xtr_ + Ptr @ config.covariate_coeffs
        Xte = Xte_ + Pte @ config.covariate_coeffs

        ytr = weights.weight_contr(Xtr)
        yte = weights.weight_contr(Xte)

        Rtr = weights.weight_contr(Ptr)
        Rte = weights.weight_contr(Pte)

        data = Dataset(
            Xtr,
            Xte,
            ytr,
            yte,
            _start_date=config.start_date,
            Ptr=Ptr,
            Pte=Pte,
            Rtr=Rtr,
            Rte=Rte,
        )
    else:
        Xtr = obs[: config.n_pre_intervention_timepoints, :]
        Xte = obs[config.n_pre_intervention_timepoints :, :]

        ytr = weights.weight_contr(Xtr)
        yte = weights.weight_contr(Xte)

        data = Dataset(Xtr, Xte, ytr, yte, _start_date=config.start_date)

    return data
