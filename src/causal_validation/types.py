import typing as tp

import numpy as np
from scipy.stats._distn_infrastructure import (
    rv_continuous,
    rv_discrete,
)

EffectTypes = tp.Literal["fixed", "random"]
WeightTypes = tp.Literal["uniform", "non-uniform"]
InterventionTypes = tp.Literal["pre-intervention", "post-intervention", "both"]
RandomVariable = tp.Union[rv_continuous, rv_discrete]
Number = tp.Union[float, int]
NPArray = np.ndarray
