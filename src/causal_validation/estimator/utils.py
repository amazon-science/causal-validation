from dataclasses import dataclass
import typing as tp

from azcausal.core.error import Error
from azcausal.core.estimator import Estimator
from azcausal.core.panel import Panel, CausalPanel
from azcausal.core.result import Result as _Result
from jaxtyping import Float
import pandas as pd

from causal_validation.data import Dataset
from causal_validation.estimator import Result
from causal_validation.types import NPArray
from azcausal.util import to_panels

def to_azcausal(dataset: Dataset) -> Panel:
    if dataset.n_treated_units != 1:
        raise ValueError("Only one treated unit is supported.")
    time_index = dataset.full_index
    unit_cols = dataset._get_columns()
    
    data = []
    for time_idx in range(dataset.n_timepoints):
        for unit_idx, unit in enumerate(unit_cols):
            data.append({
                'variable': unit,
                'time': time_index[time_idx],
                'value': dataset.Y[time_idx, unit_idx],
                'treated': int(dataset.D[time_idx, unit_idx])
            })
    
    df_data =  pd.DataFrame(data)
    panels = to_panels(df_data, "time", "variable", ["value", "treated"])
    ctypes = dict(
        outcome="value", time="time", unit="variable", intervention="treated"
    )
    panel = CausalPanel(panels).setup(**ctypes)
    return panel


@dataclass
class AZCausalWrapper:
    model: Estimator
    error_estimator: tp.Optional[Error] = None
    _az_result: _Result = None

    def __post_init__(self):
        self._model_name = self.model.__class__.__name__

    def __call__(self, data: Dataset, **kwargs) -> Result:
        panel = to_azcausal(data)
        result = self.model.fit(panel, **kwargs)
        if self.error_estimator:
            self.model.error(result, self.error_estimator)
        self._az_result = result

        res = Result(
            effect=result.effect,
            counterfactual=self.counterfactual,
            synthetic=self.synthetic,
            observed=self.observed,
        )
        return res

    @property
    def counterfactual(self) -> Float[NPArray, "N 1"]:
        df = self._az_result.effect.by_time
        c_factual = df.loc[:, "CF"].values.reshape(-1, 1)
        return c_factual

    @property
    def synthetic(self) -> Float[NPArray, "N 1"]:
        df = self._az_result.effect.by_time
        synth_control = df.loc[:, "C"].values.reshape(-1, 1)
        return synth_control

    @property
    def observed(self) -> Float[NPArray, "N 1"]:
        df = self._az_result.effect.by_time
        treated = df.loc[:, "T"].values.reshape(-1, 1)
        return treated
