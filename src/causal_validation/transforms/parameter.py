from dataclasses import dataclass
import typing as tp

from jaxtyping import Float
import numpy as np

from causal_validation.types import RandomVariable


@dataclass
class Parameter:
    def get_value(self, **kwargs) -> Float[np.ndarray, "..."]:
        raise NotImplementedError


@dataclass
class FixedParameter(Parameter):
    value: float

    def get_value(
        self, n_units: int, n_timepoints: int
    ) -> Float[np.ndarray, "{n_timepoints} {n_units}"]:
        return np.ones(shape=(n_timepoints, n_units)) * self.value


@dataclass
class RandomParameter(Parameter):
    sampling_dist: RandomVariable
    random_state: int = 123


@dataclass
class UnitVaryingParameter(RandomParameter):
    def get_value(
        self, n_units: int, n_timepoints: int
    ) -> Float[np.ndarray, "{n_timepoints} {n_units}"]:
        unit_param = self.sampling_dist.rvs(
            size=(n_units,), random_state=self.random_state
        )
        return np.stack([unit_param] * n_timepoints)


@dataclass
class TimeVaryingParameter(RandomParameter):
    def get_value(
        self, n_units: int, n_timepoints: int
    ) -> Float[np.ndarray, "{n_timepoints} {n_units}"]:
        time_param = self.sampling_dist.rvs(
            size=(n_timepoints, 1), random_state=self.random_state
        )
        return np.tile(time_param, reps=n_units)


ParameterOrFloat = tp.Union[Parameter, float]


def resolve_parameter(value: ParameterOrFloat) -> Parameter:
    if isinstance(value, tp.Union[int, float]):
        return FixedParameter(value=value)
    elif isinstance(value, Parameter):
        return value
    else:
        raise TypeError("`value` argument must be either a `Parameter` or `float`.")
