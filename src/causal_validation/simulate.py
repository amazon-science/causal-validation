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
    Xtr = obs[: config.n_pre_intervention_timepoints, :]
    Xte = obs[config.n_pre_intervention_timepoints :, :]
    ytr = weights.weight_obs(Xtr)
    yte = weights.weight_obs(Xte)
    data = Dataset(Xtr, Xte, ytr, yte, _start_date=config.start_date)
    return data
