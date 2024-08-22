from hypothesis import (
    given,
    strategies as st,
)

from causal_validation.effects import StaticEffect
from causal_validation.testing import (
    TestConstants,
    simulate_data,
)

EFFECT_LOWER_BOUND = 1e-3


@st.composite
def effect_strategy(draw):
    lower_range = st.floats(
        min_value=-0.1,
        max_value=-1e-4,
        exclude_max=True,
        allow_infinity=False,
        allow_nan=False,
    )
    upper_range = st.floats(
        min_value=1e-4,
        max_value=0.1,
        exclude_min=True,
        allow_infinity=False,
        allow_nan=False,
    )
    combined_strategy = st.one_of(lower_range, upper_range)
    return draw(combined_strategy)


@given(
    global_mean=st.floats(
        min_value=20.0, max_value=50.0, allow_nan=False, allow_infinity=False
    ),
    effect_val=effect_strategy(),
    seed=st.integers(min_value=1, max_value=10),
)
def test_array_shapes(global_mean: float, effect_val: float, seed: int):
    constants = TestConstants(GLOBAL_SCALE=0.01)
    data = simulate_data(global_mean, seed, constants=constants)
    effect = StaticEffect(effect=effect_val)

    inflated_data = effect(data)
    if effect_val == 0:
        assert inflated_data.yte.sum() == data.yte.sum()
    elif effect_val < 0:
        assert inflated_data.yte.sum() < data.yte.sum()
    elif effect_val > 0:
        assert inflated_data.yte.sum() > data.yte.sum()

    assert inflated_data.counterfactual.sum() == data.yte.sum()

    _effects = effect.get_effect(data)
    assert _effects.shape == (data.n_post_intervention, 1)
