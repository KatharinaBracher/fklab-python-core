"""
================================
Segments (:mod:`fklab.segments`)
================================

.. currentmodule:: fklab.segments

Functions and classes to work with time segments. All functions defined
in the basic_algorithms module and the Segment class are also available
directly from the top level segments package.

"""
from .basic_algorithms import *
from .segment import Segment
from .segment import SegmentError

__all__ = [_s for _s in dir() if not _s.startswith("_")]
