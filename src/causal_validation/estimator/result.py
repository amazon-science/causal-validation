from dataclasses import dataclass

from azcausal.core.effect import Effect
from jaxtyping import Float

from causal_validation.types import NPArray


@dataclass
class Result:
    effect: Effect
    counterfactual: Float[NPArray, "N 1"]
    synthetic: Float[NPArray, "N 1"]
    observed: Float[NPArray, "N 1"]