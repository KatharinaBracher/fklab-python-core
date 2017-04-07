"""
===============================================
Digital Filtering (:mod:`fklab.signals.filter`)
===============================================

.. currentmodule:: fklab.signals.filter

Digital filtering functions.

.. automodule:: fklab.signals.filter.filter
    
"""

from .filter import *

__all__ = [s for s in dir() if not s.startswith('_')]
