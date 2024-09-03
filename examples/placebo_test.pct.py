# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: causal-validation
#     language: python
#     name: python3
# ---

# %%
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
from causal_validation.validation.placebo import PlaceboTest
from causal_validation.models import AZCausalWrapper
from azcausal.estimators.panel.did import DID
from azcausal.core.error import JackKnife

# %%
cfg = cv.Config(
    n_control_units=10,
    n_pre_intervention_timepoints=60,
    n_post_intervention_timepoints=30,
    seed=123,
)

TRUE_EFFECT = 0.05
effect = StaticEffect(effect=TRUE_EFFECT)
data = effect(simulate(cfg))
plot(data)

# %%
model = AZCausalWrapper(model=VTLab(), error_estimator=JackKnife())
result = PlaceboTest(model, data).execute()
result.summary()

# %%
