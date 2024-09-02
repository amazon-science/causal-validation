from hypothesis import settings
from jaxtyping import install_import_hook

settings.register_profile(
    "causal_validation", database=None, max_examples=10, deadline=None
)
settings.load_profile("causal_validation")
with install_import_hook("causal_validation", "beartype.beartype"):
    import causal_validation  # noqa: F401
