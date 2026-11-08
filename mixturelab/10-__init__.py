"""Manual constrained Gaussian-mixture density estimation."""
from .core import Model, expectation, em, score
__all__ = ['Model', 'expectation', 'em', 'score']
__version__ = '0.1.0'
