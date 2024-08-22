from dataclasses import dataclass
import typing as tp

from causal_validation.config import Config
from causal_validation.data import Dataset
from causal_validation.simulate import simulate


@dataclass(frozen=True, kw_only=True)
class TestConstants:
    N_CONTROL: int = 10
    N_PRE_TREATMENT: int = 500
    N_POST_TREATMENT: int = 500
    DATA_SLOTS: tp.Tuple[str, str, str, str] = ("Xtr", "Xte", "ytr", "yte")
    ZERO_DIVISION_ERROR: float = 1e-6
    GLOBAL_SCALE: float = 1.0
    __test__: bool = False


def simulate_data(
    global_mean: float, seed: int, constants: tp.Optional[TestConstants] = None
) -> Dataset:
    if not constants:
        constants = TestConstants()
    cfg = Config(
        n_control_units=constants.N_CONTROL,
        n_pre_intervention_timepoints=constants.N_PRE_TREATMENT,
        n_post_intervention_timepoints=constants.N_POST_TREATMENT,
        global_mean=global_mean,
        global_scale=constants.GLOBAL_SCALE,
        seed=seed,
    )
    return simulate(config=cfg)
