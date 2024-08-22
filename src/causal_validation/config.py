from dataclasses import (
    dataclass,
    field,
)
import datetime as dt
import typing as tp

import numpy as np

from causal_validation.weights import UniformWeights

if tp.TYPE_CHECKING:
    from causal_validation.types import WeightTypes


@dataclass(kw_only=True, frozen=True)
class WeightConfig:
    weight_type: "WeightTypes" = field(default_factory=UniformWeights)


@dataclass(kw_only=True)
class Config:
    n_control_units: int
    n_pre_intervention_timepoints: int
    n_post_intervention_timepoints: int
    global_mean: float = 20.0
    global_scale: float = 0.2
    start_date: dt.date = dt.date(year=2023, month=1, day=1)
    seed: int = 123
    weights_cfg: WeightConfig = field(default_factory=WeightConfig)

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)
