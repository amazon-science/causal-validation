from copy import deepcopy
from dataclasses import dataclass
import typing as tp

from jaxtyping import Float
import numpy as np

from causal_validation.data import Dataset
from causal_validation.transforms.parameter import resolve_parameter

if tp.TYPE_CHECKING:
    from causal_validation.transforms.parameter import (
        Parameter,
        resolve_parameter,
    )


@dataclass(kw_only=True)
class AbstractTransform:
    _slots: tp.Optional[tp.Tuple[str]] = None

    def __post_init__(self):
        if self._slots:
            for slot in self._slots:
                coerced_param = resolve_parameter(getattr(self, slot))
                setattr(self, slot, coerced_param)

    def __call__(self, data: Dataset) -> Dataset:
        transform_vals = self.get_values(data)
        return self.apply_values(transform_vals, data=data)

    def get_values(self, data: Dataset) -> Float[np.ndarray, "T N"]:
        raise NotImplementedError

    def apply_values(
        self,
        transform_vals: np.ndarray,
        data: Dataset,
    ) -> Dataset:
        raise NotImplementedError

    @staticmethod
    def _resolve_parameter(
        data: Dataset, parameter: "Parameter"
    ) -> Float[np.ndarray, "..."]:
        data_params = data._slots
        return parameter.get_value(**data_params)

    def _get_parameter_values(self, data: Dataset) -> tp.Dict[str, np.ndarray]:
        param_vals = {}
        if self._slots:
            for slot in self._slots:
                param = getattr(self, slot)
                param_vals[slot] = self._resolve_parameter(data, param)
        return param_vals


@dataclass(kw_only=True)
class AdditiveOutputTransform(AbstractTransform):
    def apply_values(
        self,
        transform_vals: np.ndarray,
        data: Dataset,
    ) -> Dataset:
        Y = deepcopy(data.Y)
        Y = Y + transform_vals
        return Dataset(Y, data.D, data.X, data._start_date, data._name)


@dataclass(kw_only=True)
class MultiplicativeOutputTransform(AbstractTransform):
    def apply_values(
        self,
        transform_vals: np.ndarray,
        data: Dataset,
    ) -> Dataset:
        Y = deepcopy(data.Y)
        Y = Y * transform_vals
        return Dataset(Y, data.D, data.X, data._start_date, data._name)


@dataclass(kw_only=True)
class AdditiveCovariateTransform(AbstractTransform):
    def apply_values(
        self,
        transform_vals: np.ndarray,
        data: Dataset,
    ) -> Dataset:
        X = deepcopy(data.X)
        X = X + transform_vals
        return Dataset(data.Y, data.D, X, data._start_date, data._name)
