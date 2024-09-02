from __future__ import annotations

from dataclasses import dataclass
import typing as tp

from jaxtyping import Float
import numpy as np

from causal_validation.base import BaseObject

if tp.TYPE_CHECKING:
    from causal_validation.config import WeightConfig


@dataclass
class AbstractWeights(BaseObject):
    name: str = "Abstract Weights"

    def _get_weights(self, obs: Float[np.ndarray, "N D"]) -> Float[np.ndarray, "D 1"]:
        raise NotImplementedError("Please implement `_get_weights` in all subclasses.")

    def get_weights(self, obs: Float[np.ndarray, "N D"]) -> Float[np.ndarray, "D 1"]:
        weights = self._get_weights(obs)

        np.testing.assert_almost_equal(
            weights.sum(), 1.0, decimal=1.0, err_msg="Weights must sum to 1."
        )
        assert min(weights >= 0), "Weights should be non-negative"
        return weights

    def __call__(self, obs: Float[np.ndarray, "N D"]) -> Float[np.ndarray, "N 1"]:
        return self.weight_obs(obs)

    def weight_obs(self, obs: Float[np.ndarray, "N D"]) -> Float[np.ndarray, "N 1"]:
        weights = self.get_weights(obs)

        weighted_obs = obs @ weights
        return weighted_obs


@dataclass
class UniformWeights(AbstractWeights):
    name: str = "Uniform Weights"

    def _get_weights(self, obs: Float[np.ndarray, "N D"]) -> Float[np.ndarray, "D 1"]:
        n_units = obs.shape[1]
        return np.repeat(1.0 / n_units, repeats=n_units).reshape(-1, 1)


def resolve_weights(config: "WeightConfig") -> AbstractWeights:
    if config.weight_type == "uniform":
        return UniformWeights()
