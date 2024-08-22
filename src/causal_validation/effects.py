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

    def get_effect(self, data: Dataset, **kwargs) -> Float[np.ndarray, "N 1"]:
        raise NotImplementedError("Please implement `get_effect` in all subclasses.")

    def __call__(self, data: Dataset, **kwargs) -> Dataset:
        inflation_vals = self.get_effect(data)
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
    effect: float
    name = "Static Effect"

    def get_effect(self, data: Dataset, **kwargs) -> Float[np.ndarray, "N 1"]:
        n_post_intervention = data.n_post_intervention
        return np.repeat(1.0 + self.effect, repeats=n_post_intervention)[:, None]


@dataclass
class RandomEffect(AbstractEffect, _RandomEffect):
    mean_effect: float
    stddev_effect: float
    name: str = "Random Effect"

    def get_effect(
        self, data: Dataset, key: np.random.RandomState
    ) -> Float[np.ndarray, "N 1"]:
        n_post_intervention = data.n_post_intervention
        effect_sample = key.normal(
            loc=1.0 + self.mean_effect,
            scale=self.stddev_effect,
            size=(n_post_intervention, 1),
        )
        return effect_sample


# Placeholder for now.
def resolve_effect(cfg: "EffectConfig") -> AbstractEffect:
    return StaticEffect(effect=cfg.effect)
