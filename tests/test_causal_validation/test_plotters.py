from hypothesis import (
    given,
    settings,
    strategies as st,
)
from matplotlib.axes._axes import Axes
import matplotlib.pyplot as plt

from causal_validation.plotters import plot
from causal_validation.testing import (
    TestConstants,
    simulate_data,
)

DEFAULT_SEED = 123
NUM_AUX_LINES = 2
LARGE_N_POST = 5000
LARGE_N_PRE = 5000
N_LEGEND_ENTRIES = 3


# Define a strategy for generating titles
title_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "
    ),
    min_size=1,
    max_size=50,
)


@given(
    n_control=st.integers(min_value=1, max_value=10),
    n_pre_treatment=st.integers(min_value=1, max_value=50),
    n_post_treatment=st.integers(min_value=1, max_value=50),
    ax_bool=st.booleans(),
)
@settings(max_examples=10)
def test_plot(
    n_control: int, n_pre_treatment: int, n_post_treatment: int, ax_bool: bool
):
    constants = TestConstants(
        N_POST_TREATMENT=n_post_treatment,
        N_PRE_TREATMENT=n_pre_treatment,
        N_CONTROL=n_control,
    )
    data = simulate_data(0.0, DEFAULT_SEED, constants=constants)
    if ax_bool:
        _, ax = plt.subplots()
    else:
        ax = None
    ax = plot(data, ax=ax)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == n_control + 2
    assert ax.get_legend() is not None
    assert len(ax.get_legend().get_texts()) == N_LEGEND_ENTRIES
    plt.close()
