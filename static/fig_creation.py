import matplotlib.pyplot as plt
from scipy.stats import norm

from causal_validation import (
    Config,
    simulate,
)
from causal_validation.effects import StaticEffect
from causal_validation.plotters import plot
from causal_validation.transforms import (
    Periodic,
    Trend,
)
from causal_validation.transforms.parameter import UnitVaryingParameter

plt.style.use("style.mplstyle")

if __name__ == "__main__":
    cfg = Config(
        n_control_units=10,
        n_pre_intervention_timepoints=60,
        n_post_intervention_timepoints=30,
    )

    # Simulate the base observation
    base_data = simulate(cfg)

    # Apply a linear trend with unit-varying intercept
    intercept = UnitVaryingParameter(sampling_dist=norm(0, 1))
    trend_component = Trend(degree=1, coefficient=0.1, intercept=intercept)
    trended_data = trend_component(base_data)

    # Simulate a 5% lift in the treated unit's post-intervention data
    effect = StaticEffect(0.05)
    inflated_data = effect(trended_data)
    plot(inflated_data)
    plt.savefig("readme_fig.png", dpi=150)
