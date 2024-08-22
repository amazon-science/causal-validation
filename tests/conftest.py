from hypothesis import settings

settings.register_profile(
    "causal_validation", database=None, max_examples=10, deadline=None
)
settings.load_profile("causal_validation")
