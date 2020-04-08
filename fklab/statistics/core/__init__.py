"""
====================================================
Statistical functions (:mod:`fklab.statistics.core`)
====================================================

.. currentmodule:: fklab.statistics.core

Collection of statistical functions.


"""
from .bootstrap import *
from .general import *

__all__ = [s for s in dir() if not s.startswith("_")]
