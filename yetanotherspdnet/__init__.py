"""Yet Another SPDNet - A robust and tested implementation of SPDNet learning models."""

__version__ = "0.1.0"

# Import main modules
from yetanotherspdnet import functions, nn, random
from yetanotherspdnet.models import SPDnet
from yetanotherspdnet.meta_models import create_model


# Define public API
__all__ = [
    "functions",
    "nn",
    "random",
    "SPDnet",
    "create_model",
    "__version__",
]
