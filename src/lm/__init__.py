"""Top-level package for lm."""

__author__ = """Fabrizio Milo"""
__email__ = "remove-this-fmilo@entropysource.com"
__version__ = "0.1.0"


from .registry import register_dataset, register_infeed, register_model, register_encoder

__all__ = [
    "register_model",
    "register_dataset",
    "register_encoder",
    "register_infeed",
    "__version__",
]
