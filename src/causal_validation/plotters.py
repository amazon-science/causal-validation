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
    Y_control = data.control_unit_outputs
    Y_treated = data.treated_unit_outputs
    idx = data.full_index
    
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
    
    ax.plot(idx, Y_control, color=cols[0], label="Control", alpha=0.5)
    
    for i, unit_idx in enumerate(data.treated_unit_indices):
        unit_color = cols[1] if len(data.treated_unit_indices) == 1 else cols[1 + i % (len(cols) - 2)]
        unit_label = "Treated" if len(data.treated_unit_indices) == 1 else f"Treated {unit_idx}"
        ax.plot(idx, Y_treated[:, i], color=unit_color, label=unit_label)
        
        treatment_date = data.treatment_date(unit_idx)
        if treatment_date is not None:
            line_color = cols[2] if len(data.treated_unit_indices) == 1 else unit_color
            line_label = "Intervention" if len(data.treated_unit_indices) == 1 else f"Intervention {unit_idx}"
            ax.axvline(x=treatment_date, color=line_color, label=line_label, linestyle="--", alpha=0.7)
    
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )
    clean_legend(ax)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set(xlabel="Time", ylabel="Observed", title=title)
    return ax
