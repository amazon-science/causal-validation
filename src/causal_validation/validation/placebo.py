from dataclasses import dataclass
import typing as tp

from azcausal.core.effect import Effect
import numpy as np
import pandas as pd
from pandera import (
    Check,
    Column,
    DataFrameSchema,
)
from rich import box
from rich.table import Table
from scipy.stats import ttest_1samp

from causal_validation.data import Dataset
from causal_validation.models import AZCausalWrapper
from rich.progress import track, ProgressBar, Progress

PlaceboSchema = DataFrameSchema(
    {
        "Model": Column(str),
        "Effect": Column(float, coerce=True),
        "Standard Deviation": Column(
            float, checks=[Check.greater_than(0.0)], coerce=True
        ),
        "Standard Error": Column(float, checks=[Check.greater_than(0.0)], coerce=True),
        "p-value": Column(float, coerce=True),
    }
)


@dataclass
class PlaceboTestResult:
    effects: tp.Dict[str, tp.List[Effect]]

    def _model_to_df(self, model_name: str, effects: tp.List[Effect]) -> pd.DataFrame:
        _effects = [effect.value for effect in effects]
        _n_effects = len(_effects)
        expected_effect = np.mean(_effects)
        stddev_effect = np.std(_effects)
        std_error = stddev_effect / np.sqrt(_n_effects)
        p_value = ttest_1samp(_effects, 0, alternative="two-sided").pvalue
        result = {
            "Model": model_name,
            "Effect": expected_effect,
            "Standard Deviation": stddev_effect,
            "Standard Error": std_error,
            "p-value": p_value,
        }
        result_df = pd.DataFrame([result])
        return result_df

    def to_df(self) -> pd.DataFrame:
        df = pd.concat(
            [
                self._model_to_df(model, effects)
                for model, effects in self.effects.items()
            ]
        )
        PlaceboSchema.validate(df)
        return df

    def summary(self, precision: int = 4) -> Table:
        table = Table(show_header=True, box=box.MARKDOWN)
        df = self.to_df()
        numeric_cols = df.select_dtypes(include=[np.number])
        df.loc[:, numeric_cols.columns] = np.round(numeric_cols, decimals=precision)

        for column in df.columns:
            table.add_column(str(column), style="magenta")

        for _, value_list in enumerate(df.values.tolist()):
            row = [str(x) for x in value_list]
            table.add_row(*row)

        return table


@dataclass
class PlaceboTest:
    models: tp.Union[AZCausalWrapper, tp.List[AZCausalWrapper]]
    dataset: Dataset

    def __post_init__(self):
        if isinstance(self.models, AZCausalWrapper):
            self.models: tp.List[AZCausalWrapper] = [self.models]
        self._n_control_units = self.dataset.n_units

    def execute(self, verbose: bool = True) -> PlaceboTestResult:
        results = {}
        # model_looper, unit_looper = self._get_progress_bars(verbose)
        disable = not verbose
        with Progress() as progress:
            model_task = progress.add_task(
                "[red]Models...", total=len(self.models), visible=disable
            )
            unit_task = progress.add_task(
                "[green]Units...", total=self._n_control_units, visible=disable
            )
            for model in self.models:
                model_result = []
                progress.update(model_task, advance=1)
                for i in range(self._n_control_units):
                    progress.update(unit_task, advance=1)
                    placebo_data = self.dataset.to_placebo_data(i)
                    result = model(placebo_data)
                    result = result.effect.percentage()
                    model_result.append(result)
                results[model._model_name] = model_result
        return PlaceboTestResult(effects=results)

    def _get_progress_bars(self, verbose: bool) -> tp.Tuple[ProgressBar, ProgressBar]:
        if verbose:
            model_looper = track(self.models, description="Model cycling")
            unit_looper = track(
                range(self._n_control_units), description="Unit cycling"
            )
        else:
            model_looper = track(self.models, description="Model cycling", disable=True)
            unit_looper = track(
                range(self._n_control_units), description="Unit cycling", disable=True
            )
        return model_looper, unit_looper
