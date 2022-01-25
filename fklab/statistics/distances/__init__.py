"""
====================================================
Distance metrics (:mod:`fklab.statistics.distances`)
====================================================

.. currentmodule:: fklab.statistics.distances

Functions to compute distance metrics.

"""
from .distances import *

__all__ = [s for s in dir() if not s.startswith("_")]
