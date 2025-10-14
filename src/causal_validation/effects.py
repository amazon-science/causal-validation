from dataclasses import dataclass
import typing as tp

from jaxtyping import Float
import numpy as np

from causal_validation.base import BaseObject
from causal_validation.data import Dataset

if tp.TYPE_CHECKING:
    from causal_validation.config import EffectConfig


@dataclass
class AbstractEffect(BaseObject):
    name: str = "Abstract Effect"

    def get_effect(self, data: Dataset, **kwargs) -> Float[np.ndarray, "T N"]:
        raise NotImplementedError("Please implement `get_effect` in all subclasses.")

    def __call__(self, data: Dataset, **kwargs) -> Dataset:
        inflation_vals = self.get_effect(data, **kwargs)
        return data.inflate(inflation_vals)


@dataclass
class _StaticEffect:
    effect: float


@dataclass
class _RandomEffect:
    mean_effect: float
    stddev_effect: float


@dataclass
class StaticEffect(AbstractEffect, _StaticEffect):
    """
    Static effect to be applied on treated units in treatment periods.
    The effect is meant to be applied proportionally to the treatment
    dosage.

    Attributes:
        effect (float): Rate effect to be applied, i.e., 0.3 = 30% lift.
        name (str): Name for the effect. 'Static Effect' by default.
    """
    effect: float
    name: str = "Static Effect"

    def get_effect(self, data: Dataset, **kwargs) -> Float[np.ndarray, "T N"]:
        return np.ones(data.D.shape) + data.D*self.effect


@dataclass
class RandomEffect(AbstractEffect, _RandomEffect):
    """
    Random effect to be applied on treated units in treatment periods.
    The effect is meant to be applied proportionally to the treatment
    dosage. The effects are randomly sampled from Normal dist.

    Attributes:
        mean_effect (float): Rate effect mean to be applied.
        stddev_effect (float): Rate effect std. dev. to be applied.
        name (str): Name for the effect. 'Random Effect' by default.
    """
    mean_effect: float
    stddev_effect: float
    name: str = "Random Effect"

    def get_effect(
        self, data: Dataset, key: np.random.RandomState
    ) -> Float[np.ndarray, "T N"]:
        effect_sample = key.normal(
            loc=self.mean_effect,
            scale=self.stddev_effect,
            size=data.D.shape
        )
        return np.ones(data.D.shape) + data.D*effect_sample


# Placeholder for now.
def resolve_effect(cfg: "EffectConfig") -> AbstractEffect:
    return StaticEffect(effect=cfg.effect)
