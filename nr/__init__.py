"""Traditional (non-learning) speech noise reduction."""
from .config import DEFAULT, Config, load_config
from .methods import METHODS, enhance

__all__ = ["Config", "DEFAULT", "load_config", "METHODS", "enhance"]
