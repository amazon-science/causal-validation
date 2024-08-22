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
        vals = self.get_values(data)
        pre_intervention_trend = vals[: data.n_pre_intervention]
        post_intervention_trend = vals[data.n_pre_intervention :]
        return self.apply_values(
            pre_intervention_trend, post_intervention_trend, data=data
        )

    def get_values(self, data: Dataset) -> Float[np.ndarray, "N D"]:
        raise NotImplementedError

    def apply_values(
        self,
        pre_intervention_vals: np.ndarray,
        post_intervention_vals: np.ndarray,
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
class AdditiveTransform(AbstractTransform):
    def apply_values(
        self,
        pre_intervention_vals: np.ndarray,
        post_intervention_vals: np.ndarray,
        data: Dataset,
    ) -> Dataset:
        Xtr, ytr = [deepcopy(i) for i in data.pre_intervention_obs]
        Xte, yte = [deepcopy(i) for i in data.post_intervention_obs]
        Xtr = Xtr + pre_intervention_vals[:, 1:]
        ytr = ytr + pre_intervention_vals[:, :1]
        Xte = Xte + post_intervention_vals[:, 1:]
        yte = yte + post_intervention_vals[:, :1]
        return Dataset(Xtr, Xte, ytr, yte, data._start_date, data.counterfactual)


@dataclass(kw_only=True)
class MultiplicativeTransform(AbstractTransform):
    def apply_values(
        self,
        pre_intervention_vals: np.ndarray,
        post_intervention_vals: np.ndarray,
        data: Dataset,
    ) -> Dataset:
        Xtr, ytr = [deepcopy(i) for i in data.pre_intervention_obs]
        Xte, yte = [deepcopy(i) for i in data.post_intervention_obs]
        Xtr = Xtr * pre_intervention_vals
        ytr = ytr * pre_intervention_vals
        Xte = Xte * post_intervention_vals
        yte = yte * post_intervention_vals
        return Dataset(Xtr, Xte, ytr, yte, data._start_date, data.counterfactual)
