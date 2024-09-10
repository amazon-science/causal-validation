from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import datetime as dt
import typing as tp

from azcausal.core.panel import CausalPanel
from azcausal.util import to_panels
from jaxtyping import (
    Float,
    Integer,
)
import numpy as np
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from pandas.core.indexes.datetimes import DatetimeIndex

from causal_validation.types import InterventionTypes


@dataclass
class Dataset:
    Xtr: Float[np.ndarray, "N D"]
    Xte: Float[np.ndarray, "M D"]
    ytr: Float[np.ndarray, "N 1"]
    yte: Float[np.ndarray, "M 1"]
    _start_date: dt.date
    counterfactual: tp.Optional[Float[np.ndarray, "M 1"]] = None
    _name: str = None

    def to_df(
        self, index_start: str = dt.date(year=2023, month=1, day=1)
    ) -> pd.DataFrame:
        inputs = np.vstack([self.Xtr, self.Xte])
        outputs = np.vstack([self.ytr, self.yte])
        data = np.hstack([outputs, inputs])
        index = self._get_index(index_start)
        colnames = self._get_columns()
        indicator = self._get_indicator()
        df = pd.DataFrame(data, index=index, columns=colnames)
        df = df.assign(treated=indicator)
        return df

    @property
    def n_post_intervention(self) -> int:
        return self.Xte.shape[0]

    @property
    def n_pre_intervention(self) -> int:
        return self.Xtr.shape[0]

    @property
    def n_units(self) -> int:
        return self.Xtr.shape[1]

    @property
    def n_timepoints(self) -> int:
        return self.n_post_intervention + self.n_pre_intervention

    @property
    def control_units(self) -> Float[np.ndarray, "{self.n_timepoints} {self.n_units}"]:
        return np.vstack([self.Xtr, self.Xte])

    @property
    def treated_units(self) -> Float[np.ndarray, "{self.n_timepoints} 1"]:
        return np.vstack([self.ytr, self.yte])

    @property
    def pre_intervention_obs(
        self,
    ) -> tp.Tuple[Float[np.ndarray, "N D"], Float[np.ndarray, "N 1"]]:
        return self.Xtr, self.ytr

    @property
    def post_intervention_obs(
        self,
    ) -> tp.Tuple[Float[np.ndarray, "M D"], Float[np.ndarray, "M 1"]]:
        return self.Xte, self.yte

    @property
    def full_index(self) -> DatetimeIndex:
        return self._get_index(self._start_date)

    @property
    def treatment_date(self) -> Timestamp:
        idxs = self.full_index
        return idxs[self.n_pre_intervention]

    def get_index(self, period: InterventionTypes) -> DatetimeIndex:
        if period == "pre-intervention":
            return self.full_index[: self.n_pre_intervention]
        elif period == "post-intervention":
            return self.full_index[self.n_pre_intervention :]
        else:
            return self.full_index

    def _get_columns(self) -> tp.List[str]:
        colnames = ["T"] + [f"C{i}" for i in range(self.n_units)]
        return colnames

    def _get_index(self, start_date: dt.date) -> DatetimeIndex:
        return pd.date_range(start=start_date, freq="D", periods=self.n_timepoints)

    def _get_indicator(self) -> Integer[np.ndarray, "N 1"]:
        indicator = np.vstack(
            [
                np.zeros(shape=(self.n_pre_intervention, 1)).astype(np.int64),
                np.ones(shape=(self.n_post_intervention, 1)).astype(np.int64),
            ]
        )
        return indicator

    def inflate(self, inflation_vals: Float[np.ndarray, "M 1"]) -> Dataset:
        Xtr, ytr = [deepcopy(i) for i in self.pre_intervention_obs]
        Xte, yte = [deepcopy(i) for i in self.post_intervention_obs]
        inflated_yte = yte * inflation_vals
        return Dataset(Xtr, Xte, ytr, inflated_yte, self._start_date, yte)

    def __eq__(self, other: Dataset) -> bool:
        ytr = np.allclose(self.ytr, other.ytr)
        yte = np.allclose(self.yte, other.yte)
        if self.Xtr.shape == other.Xtr.shape:
            xtr = np.allclose(self.Xtr, other.Xtr)
        else:
            xtr = False
        if self.Xte.shape == other.Xte.shape:
            xte = np.allclose(self.Xte, other.Xte)
        else:
            xte = False
        return all([xtr, ytr, xte, yte])

    def to_azcausal(self):
        time_index = np.arange(self.n_timepoints)
        data = self.to_df().assign(time=time_index).melt(id_vars=["time", "treated"])
        data.loc[:, "treated"] = np.where(
            (data["variable"] == "T") & (data["treated"] == 1.0), 1, 0
        )
        panels = to_panels(data, "time", "variable", ["value", "treated"])
        ctypes = dict(
            outcome="value", time="time", unit="variable", intervention="treated"
        )
        panel = CausalPanel(panels).setup(**ctypes)
        return panel

    @property
    def _slots(self) -> tp.Dict[str, int]:
        return {"n_units": self.n_units + 1, "n_timepoints": self.n_timepoints}

    def drop_unit(self, idx: int) -> Dataset:
        Xtr = np.delete(self.Xtr, [idx], axis=1)
        Xte = np.delete(self.Xte, [idx], axis=1)
        return Dataset(
            Xtr, Xte, self.ytr, self.yte, self._start_date, self.counterfactual
        )

    def to_placebo_data(self, to_treat_idx: int) -> Dataset:
        ytr = self.Xtr[:, to_treat_idx].reshape(-1, 1)
        yte = self.Xte[:, to_treat_idx].reshape(-1, 1)
        dropped_data = self.drop_unit(to_treat_idx)
        placebo_data = reassign_treatment(dropped_data, ytr, yte)
        return placebo_data

    @property
    def name(self) -> tp.Optional[str]:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value


@dataclass
class DatasetContainer:
    datasets: tp.List[Dataset]
    names: tp.Optional[tp.List[str]] = None

    def __post_init__(self):
        self.n_datasets = len(self.datasets)
        if self.names is None:
            names = []
            for idx, dataset in enumerate(self.datasets):
                if dataset.name:
                    names.append(dataset.name)
                else:
                    names.append(f"Dataset {idx}")
            self.names = names

    def __iter__(self) -> tp.Iterator[Dataset]:
        return iter(self.datasets)

    def as_dict(self) -> tp.Dict[str, Dataset]:
        dict_result = {}
        for n, d in zip(self.names, self.datasets, strict=True):
            dict_result[n] = d
        return dict_result

    def __len__(self) -> int:
        return len(self.datasets)


def reassign_treatment(
    data: Dataset, ytr: Float[np.ndarray, "N 1"], yte: Float[np.ndarray, "M 1"]
) -> Dataset:
    Xtr = data.Xtr
    Xte = data.Xte
    return Dataset(Xtr, Xte, ytr, yte, data._start_date, data.counterfactual)
