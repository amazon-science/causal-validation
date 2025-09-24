from __future__ import annotations

from dataclasses import dataclass
import typing as tp

from jaxtyping import Float
import numpy as np

from causal_validation.base import BaseObject

if tp.TYPE_CHECKING:
    from causal_validation.config import WeightConfig

# Constants for array dimensions
_NDIM_2D = 2
_NDIM_3D = 3


@dataclass
class AbstractWeights(BaseObject):
    name: str = "Abstract Weights"

    def _get_weights(
        self, obs: Float[np.ndarray, "N D"] | Float[np.ndarray, "N D K"]
    ) -> Float[np.ndarray, "D 1"]:
        raise NotImplementedError("Please implement `_get_weights` in all subclasses.")

    def get_weights(
        self, obs: Float[np.ndarray, "N D"] | Float[np.ndarray, "N D K"]
    ) -> Float[np.ndarray, "D 1"]:
        weights = self._get_weights(obs)

        np.testing.assert_almost_equal(
            weights.sum(), 1.0, decimal=1.0, err_msg="Weights must sum to 1."
        )
        assert min(weights >= 0), "Weights should be non-negative"
        return weights

    def __call__(
        self, obs: Float[np.ndarray, "N D"] | Float[np.ndarray, "N D K"]
    ) -> Float[np.ndarray, "N 1"] | Float[np.ndarray, "N 1 K"]:
        return self.weight_contr(obs)

    def weight_contr(
        self, obs: Float[np.ndarray, "N D"] | Float[np.ndarray, "N D K"]
    ) -> Float[np.ndarray, "N 1"] | Float[np.ndarray, "N 1 K"]:
        weights = self.get_weights(obs)

        if obs.ndim == _NDIM_2D:
            weighted_obs = obs @ weights
        elif obs.ndim == _NDIM_3D:
            weighted_obs = np.einsum("n d k, d i -> n i k", obs, weights)

        return weighted_obs


@dataclass
class UniformWeights(AbstractWeights):
    name: str = "Uniform Weights"

    def _get_weights(
        self, obs: Float[np.ndarray, "N D"] | Float[np.ndarray, "N D K"]
    ) -> Float[np.ndarray, "D 1"]:
        n_units = obs.shape[1]
        return np.repeat(1.0 / n_units, repeats=n_units).reshape(-1, 1)


def resolve_weights(config: "WeightConfig") -> AbstractWeights:
    if config.weight_type == "uniform":
        return UniformWeights()
