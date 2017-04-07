"""
=======================================
Smoothing (:mod:`fklab.signals.smooth`)
=======================================

.. currentmodule:: fklab.signals.smooth

Smoothing functions.

.. automodule:: fklab.signals.smooth.kernelsmoothing
    
"""

from .kernelsmoothing import *

__all__ = [s for s in dir() if not s.startswith('_')]
