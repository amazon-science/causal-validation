from dataclasses import dataclass
import typing as tp

from jaxtyping import Float
import numpy as np
import pandas as pd
from pandera import (
    Check,
    Column,
    DataFrameSchema,
)
from rich import box
from rich.progress import (
    Progress,
    ProgressBar,
    track,
)

from causal_validation.validation.placebo import PlaceboTest
from causal_validation.validation.testing import (
    RMSPETestStatistic,
    TestResult,
    TestResultFrame,
)

RMSPESchema = DataFrameSchema(
    {
        "Model": Column(str),
        "Dataset": Column(str),
        "Test statistic": Column(float, coerce=True),
        "p-value": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(1.0),
            ],
            coerce=True,
        ),
    }
)


@dataclass
class RMSPETestResult(TestResultFrame):
    """
    A subclass of TestResultFrame, RMSPETestResult stores test statistics and p-value
    for the treated unit. Test statistics for pseudo treatment units are also stored.
    """

    treatment_test_results: tp.Dict[tp.Tuple[str, str], TestResult]
    pseudo_treatment_test_statistics: tp.Dict[tp.Tuple[str, str], tp.List[Float]]

    def to_df(self) -> pd.DataFrame:
        dfs = []
        for (model, dataset), test_results in self.treatment_test_results.items():
            result = {
                "Model": model,
                "Dataset": dataset,
                "Test statistic": test_results.test_statistic,
                "p-value": test_results.p_value,
            }
            df = pd.DataFrame([result])
            dfs.append(df)
        df = pd.concat(dfs)
        RMSPESchema.validate(df)
        return df


@dataclass
class RMSPETest(PlaceboTest):
    """
    A subclass of PlaceboTest calculates RMSPE as test statistic for all units.
    Given the RMSPE test stats, p-value for actual treatment is calculated.
    """

    def execute(self, verbose: bool = True) -> RMSPETestResult:
        treatment_results, pseudo_treatment_results = {}, {}
        datasets = self.dataset_dict
        n_datasets = len(datasets)
        n_control = sum([d.n_units for d in datasets.values()])
        rmspe = RMSPETestStatistic()
        with Progress(disable=not verbose) as progress:
            model_task = progress.add_task(
                "[red]Models", total=len(self.models), visible=verbose
            )
            data_task = progress.add_task(
                "[blue]Datasets", total=n_datasets, visible=verbose
            )
            unit_task = progress.add_task(
                f"[green]Treatment and Control Units",
                total=n_control + 1,
                visible=verbose,
            )
            for data_name, dataset in datasets.items():
                progress.update(data_task, advance=1)
                for model in self.models:
                    progress.update(unit_task, advance=1)
                    treatment_result = model(dataset)
                    treatment_idx = dataset.ytr.shape[0]
                    treatment_test_stat = rmspe(
                        dataset,
                        treatment_result.counterfactual,
                        treatment_result.synthetic,
                        treatment_idx,
                    )
                    progress.update(model_task, advance=1)
                    placebo_test_stats = []
                    for i in range(dataset.n_units):
                        progress.update(unit_task, advance=1)
                        placebo_data = dataset.to_placebo_data(i)
                        result = model(placebo_data)
                        placebo_test_stats.append(
                            rmspe(
                                placebo_data,
                                result.counterfactual,
                                result.synthetic,
                                treatment_idx,
                            )
                        )
                    pval_idx = 1
                    for p_stat in placebo_test_stats:
                        pval_idx += 1 if treatment_test_stat < p_stat else 0
                    pval = pval_idx / (n_control + 1)
                    treatment_results[(model._model_name, data_name)] = TestResult(
                        p_value=pval, test_statistic=treatment_test_stat
                    )
                    pseudo_treatment_results[(model._model_name, data_name)] = (
                        placebo_test_stats
                    )
        return RMSPETestResult(
            treatment_test_results=treatment_results,
            pseudo_treatment_test_statistics=pseudo_treatment_results,
        )
