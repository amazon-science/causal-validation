import abc
from dataclasses import dataclass
import typing as tp

from jaxtyping import Float
import numpy as np

from causal_validation.data import Dataset


@dataclass
class TestResult:
    p_value: float
    test_statistic: float


@dataclass
class AbstractTestStatistic:
    @abc.abstractmethod
    def _compute(
        self,
        dataset: Dataset,
        counterfactual: Float[np.ndarray, "N 1"],
        synthetic: tp.Optional[Float[np.ndarray, "M 1"]],
        treatment_index: int,
    ) -> Float:
        raise NotImplementedError

    def __call__(
        self,
        observed: Float[np.ndarray, "N 1"],
        counterfactual: Float[np.ndarray, "N 1"],
        synthetic: tp.Optional[Float[np.ndarray, "M 1"]],
        treatment_index: int,
    ) -> Float:
        return self._compute(observed, counterfactual, synthetic, treatment_index)


@dataclass
class RMSPETestStatistic(AbstractTestStatistic):
    @staticmethod
    def _compute(
        dataset: Dataset,
        counterfactual: Float[np.ndarray, "N 1"],
        synthetic: Float[np.ndarray, "N 1"],
        treatment_index: int,
    ) -> Float:
        _, pre_observed = dataset.pre_intervention_obs
        _, post_observed = dataset.post_intervention_obs
        _, post_counterfactual = RMSPETestStatistic._split_array(
            counterfactual, treatment_index
        )
        pre_synthetic, _ = RMSPETestStatistic._split_array(synthetic, treatment_index)
        pre_rmspe = RMSPETestStatistic._rmspe(pre_observed, pre_synthetic)
        post_rmspe = RMSPETestStatistic._rmspe(post_observed, post_counterfactual)
        test_statistic = post_rmspe / pre_rmspe
        return test_statistic

    @staticmethod
    def _rmspe(
        observed: Float[np.ndarray, "N 1"], generated: Float[np.ndarray, "N 1"]
    ) -> float:
        return np.sqrt(np.mean(np.square(observed - generated)))

    @staticmethod
    def _split_array(
        array: Float[np.ndarray, "N 1"], index: int
    ) -> tp.Tuple[Float[np.ndarray, "Nx 1"], Float[np.ndarray, "Ny 1"]]:
        left_split = array[:index, :]
        right_split = array[index:, :]
        return left_split, right_split
