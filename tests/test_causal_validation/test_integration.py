import numpy as np
import pytest

from causal_validation import (
    Config,
    simulate,
)
from causal_validation.data import Dataset
from causal_validation.transforms import (
    Periodic,
    Trend,
)


def _sum_data(data: Dataset) -> float:
    return data.Y.sum()


@pytest.mark.parametrize(
    "seed,e1,e2", [(123, 19794.92, 63849.93), (42, 19803.64, 63858.64)]
)
def test_end_to_end(seed: int, e1: float, e2: float):
    D = np.zeros((90,11))
    D[60:, -1] = 1
    cfg = Config(
        treatment_assignments=D
    )

    data = simulate(cfg)
    np.testing.assert_approx_equal(_sum_data(data), e1, significant=2)

    t = Trend()
    np.testing.assert_approx_equal(_sum_data(t(data)), e2, significant=2)

    p = Periodic()
    np.testing.assert_approx_equal(_sum_data(p(t(data))), e2, significant=2)
