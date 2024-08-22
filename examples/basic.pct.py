# %%
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import (
    norm,
    poisson,
)

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

# %% [markdown]
# ## Simulating a Dataset

# %% Simulating a dataset is as simple as specifying a `Config` object and [markdown]
# then invoking the `simulate` function. Once simulated, we may visualise the data
# through the `plot` function.

# %%
cfg = Config(
    n_control_units=10,
    n_pre_intervention_timepoints=60,
    n_post_intervention_timepoints=30,
    seed=123,
)

data = simulate(cfg)
plot(data)

# %% [markdown]
# ### Controlling baseline behaviour
#
# We observe that we have 10 control units, each of which were sampled from a Gaussian
# distribution with mean 20 and scale 0.2. Had we wished for our underlying observations
# to have more or less noise, or to have a different global mean, then we can simply
# specify that through the config file.

# %%
means = [10, 50]
scales = [0.1, 0.5]

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 6), tight_layout=True)
for (m, s), ax in zip(product(means, scales), axes.ravel()):
    cfg = Config(
        n_control_units=10,
        n_pre_intervention_timepoints=60,
        n_post_intervention_timepoints=30,
        global_mean=m,
        global_scale=s,
    )
    data = simulate(cfg)
    plot(data, ax=ax, title=f"Mean: {m}, Scale: {s}")

# %% [markdown]
# ### Reproducibility
#
# In the above four panels, we can see that whilst the mean and scale of the underlying
# data generating process is varying, the functional form of the data is the same. This
# is by design to ensure that data sampling is reproducible. To sample a new dataset,
# you may either change the underlying seed in the config file.

# %%
cfg = Config(
    n_control_units=10,
    n_pre_intervention_timepoints=60,
    n_post_intervention_timepoints=30,
    seed=42,
)

# %% [markdown]
# Reusing the same config file across simulations

# %%
fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
for ax in axes:
    data = simulate(cfg)
    plot(data, ax=ax)

# %% [markdown]
# Or manually specifying and passing your own pseudorandom number generator key

# %%

rng = np.random.RandomState(42)

fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
for ax in axes:
    data = simulate(cfg, key=rng)
    plot(data, ax=ax)

# %% [markdown]
# ### Simulating an effect
#
# In the data we have seen up until now, the treated unit has been drawn from the same
# data generating process as the control units. However, it can be helpful to also
# inflate the treated unit to observe how well our model can recover the the true
# treatment effect. To do this, we simply compose our dataset with an `Effect` object.
# In the below, we shall inflate our data by 2%.

# %%
effect = StaticEffect(effect=0.02)
inflated_data = effect(data)
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 3))
plot(data, ax=ax0, title="Original data")
plot(inflated_data, ax=ax1, title="Inflated data")

# %% [markdown]
# ### More complex generation processes
#
# The example presented above shows a very simple stationary data generation process.
# However, we may make our example more complex by including a non-stationary trend to
# the data.

# %%
trend_term = Trend(degree=1, coefficient=0.1)
data_with_trend = effect(trend_term(data))
plot(data_with_trend)

# %%
trend_term = Trend(degree=2, coefficient=0.0025)
data_with_trend = effect(trend_term(data))
plot(data_with_trend)

# %% [markdown]
# We may also include periodic components in our data

# %%
periodicity = Periodic(amplitude=2, frequency=6)
perioidic_data = effect(periodicity(trend_term(data)))
plot(perioidic_data)

# %% [markdown]
# ### Unit-level parameterisation

# %%
sampling_dist = norm(0.0, 1.0)
intercept = UnitVaryingParameter(sampling_dist=sampling_dist)
trend_term = Trend(degree=1, intercept=intercept, coefficient=0.1)
data_with_trend = effect(trend_term(data))
plot(data_with_trend)

# %%
sampling_dist = poisson(2)
frequency = UnitVaryingParameter(sampling_dist=sampling_dist)

p = Periodic(frequency=frequency)
plot(p(data))

# %% [markdown]
# ## Conclusions
#
# In this notebook we have shown how one can define their model's true underlying data
# generating process, starting from simple white-noise samples through to more complex
# example with periodic and temporal components, perhaps containing unit-level
# variation. In a follow-up notebook, we show how these datasets may be integrated with
# Amazon's own AZCausal library to compare the effect estimated by a model with the true
# effect of the underlying data generating process. A link to this notebook is
# [here](https://github.com/amazon-science/causal-validation/blob/main/examples/azcausal.pct.py).
