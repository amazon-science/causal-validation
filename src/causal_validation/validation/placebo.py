from copy import deepcopy
from dataclasses import dataclass
import typing as tp

import numpy as np
import pandas as pd
from pandera import (
    Check,
    Column,
    DataFrameSchema,
)
from rich.progress import (
    Progress,
)
from scipy.stats import ttest_1samp

from causal_validation.data import (
    Dataset,
    DatasetContainer,
)
from causal_validation.estimator import Result
from causal_validation.estimator.utils import AZCausalWrapper
from causal_validation.validation.testing import TestResultFrame

PlaceboSchema = DataFrameSchema(
    {
        "Model": Column(str),
        "Dataset": Column(str),
        "Effect": Column(float, coerce=True),
        "Standard Deviation": Column(
            float, checks=[Check.greater_than(0.0)], coerce=True
        ),
        "Standard Error": Column(float, checks=[Check.greater_than(0.0)], coerce=True),
        "p-value": Column(float, coerce=True),
    }
)


@dataclass
class PlaceboTestResult(TestResultFrame):
    effects: tp.Dict[tp.Tuple[str, str], tp.List[Result]]

    def _model_to_df(
        self, model_name: str, dataset_name: str, effects: tp.List[Result]
    ) -> pd.DataFrame:
        _effects = [e.effect.percentage().value for e in effects]
        _n_effects = len(_effects)
        expected_effect = np.mean(_effects)
        stddev_effect = np.std(_effects)
        std_error = stddev_effect / np.sqrt(_n_effects)
        p_value = ttest_1samp(_effects, 0, alternative="two-sided").pvalue
        result = {
            "Model": model_name,
            "Dataset": dataset_name,
            "Effect": expected_effect,
            "Standard Deviation": stddev_effect,
            "Standard Error": std_error,
            "p-value": p_value,
        }
        result_df = pd.DataFrame([result])
        return result_df

    def to_df(self) -> pd.DataFrame:
        dfs = []
        for (model, dataset), effect in self.effects.items():
            df = self._model_to_df(model, dataset, effect)
            dfs.append(df)
        df = pd.concat(dfs)
        PlaceboSchema.validate(df)
        return df


@dataclass
class PlaceboTest:
    models: tp.Union[AZCausalWrapper, tp.List[AZCausalWrapper]]
    datasets: tp.Union[Dataset, tp.List[Dataset], DatasetContainer]

    def __post_init__(self):
        if isinstance(self.models, AZCausalWrapper):
            self.models: tp.List[AZCausalWrapper] = [self.models]

        if isinstance(self.datasets, Dataset):
            _datasets = DatasetContainer([self.datasets])
        elif isinstance(self.datasets, list):
            _datasets = DatasetContainer(self.datasets)
        elif isinstance(self.datasets, DatasetContainer):
            _datasets = self.datasets
        self.dataset_dict: tp.Dict[str, Dataset] = _datasets.as_dict()

    def execute(self, verbose: bool = True) -> PlaceboTestResult:
        results = {}
        datasets = self.dataset_dict
        n_datasets = len(datasets)
        n_control = sum([d.n_units for d in datasets.values()])
        with Progress(disable=not verbose) as progress:
            model_task = progress.add_task(
                "[red]Models", total=len(self.models), visible=verbose
            )
            data_task = progress.add_task(
                "[blue]Datasets", total=n_datasets, visible=verbose
            )
            unit_task = progress.add_task(
                "[green]Control Units",
                total=n_control,
                visible=verbose,
            )
            for data_name, dataset in datasets.items():
                progress.update(data_task, advance=1)
                for model in self.models:
                    progress.update(model_task, advance=1)
                    model_result = []
                    for i in dataset.control_unit_indices:
                        progress.update(unit_task, advance=1)
                        placebo_data = self.to_placebo_data(dataset, i)
                        result = model(placebo_data)
                        model_result.append(result)
                    results[(model._model_name, data_name)] = model_result
        return PlaceboTestResult(effects=results)
    
    @staticmethod
    def to_placebo_data(dataset:Dataset, to_treat_idx: int) -> Dataset:
        if not dataset.n_treated_units == 1:
            raise ValueError('Placebo test is supported only with one treated unit.')
        
        Y = deepcopy(dataset.Y)
        D = deepcopy(dataset.D)
        X = deepcopy(dataset.X)

        actual_treated_idx = dataset.treated_unit_indices[0]
        d_treated = D[:, actual_treated_idx]
        D[:, to_treat_idx] = d_treated

        Y_ = np.delete(Y, [actual_treated_idx], axis=1)
        D_ = np.delete(D, [actual_treated_idx], axis=1)
        X_ = np.delete(X, [actual_treated_idx], axis=1) if X is not None else None

        return Dataset(
            Y_, D_, X_, dataset._start_date, dataset._name
        )

