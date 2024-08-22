import typing as tp

import matplotlib as mpl
from matplotlib.axes._axes import Axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from causal_validation.data import Dataset


def clean_legend(ax: Axes) -> Axes:
    """Remove duplicate legend entries from a plot.

    Args:
        ax (Axes): The matplotlib axes containing the legend to be formatted.

    Returns:
        Axes: The cleaned matplotlib axes.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="best")
    return ax


def plot(
    data: Dataset,
    ax: tp.Optional[Axes] = None,
    title: tp.Optional[str] = None,
) -> Axes:
    cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    X = data.control_units
    y = data.treated_units
    idx = data.full_index
    treatment_date = data.treatment_date

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
    ax.plot(idx, X, color=cols[0], label="Control", alpha=0.5)
    ax.plot(idx, y, color=cols[1], label="Treated")
    ax.axvline(x=treatment_date, color=cols[2], label="Intervention", linestyle="--")
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )
    clean_legend(ax)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set(xlabel="Time", ylabel="Observed", title=title)
    return ax
