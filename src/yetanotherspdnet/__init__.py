"""Yet Another SPDNet - A robust and tested implementation of SPDNet learning models."""

__version__ = "0.1.0"

# Import main modules
from yetanotherspdnet import functions, nn, random
from yetanotherspdnet.model import GBWBNRResNet, RResNet, SPDnet


# Define public API
__all__ = [
    "GBWBNRResNet",
    "RResNet",
    "SPDnet",
    "__version__",
    "functions",
    "nn",
    "random",
]
