"""Yet Another SPDNet - A robust and tested implementation of SPDNet learning models."""

__version__ = "0.1.0"

# Import main modules
from yetanotherspdnet import functions, nn, random
from yetanotherspdnet.models import SPDNet


# Define public API
__all__ = [
    "functions",
    "nn",
    "random",
    "SPDNet",
    "__version__",
]
