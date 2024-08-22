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


@dataclass(frozen=True)
class Dataset:
    Xtr: Float[np.ndarray, "N D"]
    Xte: Float[np.ndarray, "M D"]
    ytr: Float[np.ndarray, "N 1"]
    yte: Float[np.ndarray, "M 1"]
    _start_date: dt.date
    counterfactual: tp.Optional[Float[np.ndarray, "M 1"]] = None

    def to_df(self, index_start: str = "2023-01-01") -> pd.DataFrame:
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
    def control_units(self) -> Float[np.ndarray, "N+M 1"]:
        return np.vstack([self.Xtr, self.Xte])

    @property
    def treated_units(self) -> Float[np.ndarray, "N+M 1"]:
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

    def _get_index(self, start_date: str) -> pd.Series:
        return pd.date_range(start=start_date, freq="D", periods=self.n_timepoints)

    def _get_indicator(self) -> Integer[np.ndarray, "N 1"]:
        indicator = np.vstack(
            [
                np.zeros(shape=(self.n_pre_intervention, 1)),
                np.ones(shape=(self.n_post_intervention, 1)),
            ]
        )
        return indicator

    def inflate(self, inflation_vals: Float[np.ndarray, "M 1"]) -> "Dataset":
        Xtr, ytr = [deepcopy(i) for i in self.pre_intervention_obs]
        Xte, yte = [deepcopy(i) for i in self.post_intervention_obs]
        inflated_yte = yte * inflation_vals
        return Dataset(Xtr, Xte, ytr, inflated_yte, self._start_date, yte)

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
