from dataclasses import dataclass
from jaxtyping import Float
import typing as tp
import numpy as np

from causal_validation.config import Config
from causal_validation.data import Dataset
from causal_validation.simulate import simulate

from causal_validation.types import (
    Number,
    TreatedSimulationTypes,
)


@dataclass(kw_only=True)
class TestConstants:
    TREATMENT_ASSIGNMENTS: tp.Optional[Float[np.ndarray, "T N"]] = None
    TREATED_SIMULATION_TYPE: "TreatedSimulationTypes" = "control-weighted"
    DIRICHLET_CONCENTRATION: Number = 1.0
    N_COVARIATES: tp.Optional[int] = 2
    DATA_SLOTS: tp.Tuple[str, str, str] = ("Y", "D", "X")
    ZERO_DIVISION_ERROR: float = 1e-6
    GLOBAL_SCALE: float = 1.0
    __test__: bool = False

    def __post_init__(self):
        if self.TREATMENT_ASSIGNMENTS is None:
            D = np.zeros((10,5))
            D[6:, 2] = 1
            D[8:, 3] = 1
            self.TREATMENT_ASSIGNMENTS = D


def simulate_data(
    global_mean: float, seed: int, constants: tp.Optional[TestConstants] = None
) -> Dataset:
    if not constants:
        constants = TestConstants()

    cfg = Config(
        treatment_assignments=constants.TREATMENT_ASSIGNMENTS,
        treated_simulation_type=constants.TREATED_SIMULATION_TYPE,
        dirichlet_concentration=constants.DIRICHLET_CONCENTRATION,
        n_covariates=constants.N_COVARIATES,
        global_mean=global_mean,
        global_scale=constants.GLOBAL_SCALE,
        seed=seed,
    )
    return simulate(config=cfg)
