from dataclasses import dataclass
import typing as tp

from azcausal.core.effect import Effect

from causal_validation.data import Dataset
from causal_validation.models import AZCausalWrapper
import pandas as pd
import numpy as np
from pandera import Check, Column, DataFrameSchema
from tqdm import trange


PlaceboSchema = DataFrameSchema(
    {
        "Effect": Column(float, coerce=True),
        "Standard Deviation": Column(
            float, checks=[Check.greater_than(0.0)], coerce=True
        ),
        "Standard Error": Column(float, checks=[Check.greater_than(0.0)], coerce=True),
    }
)


@dataclass
class PlaceboTestResult:
    effects: tp.List[Effect]

    def summary(self) -> pd.DataFrame:
        _effects = [effect.value for effect in self.effects]
        _n_effects = len(_effects)
        expected_effect = np.mean(_effects)
        stddev_effect = np.std(_effects)
        std_error = stddev_effect / np.sqrt(_n_effects)
        result = {
            "Effect": expected_effect,
            "Standard Deviation": stddev_effect,
            "Standard Error": std_error,
        }
        result_df = pd.DataFrame([result])
        PlaceboSchema.validate(result_df)
        return result_df


@dataclass
class PlaceboTest:
    model: AZCausalWrapper
    dataset: Dataset

    def execute(self) -> PlaceboTestResult:
        n_control_units = self.dataset.n_units
        results = []
        for i in trange(n_control_units):
            placebo_data = self.dataset.to_placebo_data(i)
            result = self.model(placebo_data)
            result = result.effect.percentage()
            results.append(result)
        return PlaceboTestResult(effects=results)
