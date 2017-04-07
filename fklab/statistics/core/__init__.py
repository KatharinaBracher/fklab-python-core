"""
====================================================
Statistical functions (:mod:`fklab.statistics.core`)
====================================================

.. currentmodule:: fklab.statistics.core

Collection of statistical functions.

.. automodule:: fklab.statistics.core.bootstrap

"""

from .bootstrap import *

__all__ = [s for s in dir() if not s.startswith('_')]
