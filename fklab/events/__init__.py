"""
========================================
Event time vectors (:mod:`fklab.events`)
========================================

.. currentmodule:: fklab.events

Functions and classes to work with event time vectors.
"""
from .basic_algorithms import *
from .event import *

__all__ = [_s for _s in dir() if not _s.startswith("_")]
