"""
=================================================
Common File IO Utilities (:mod:`fklab.io.common`)
=================================================

.. currentmodule:: fklab.io.common

Common file import and export utilities.

"""
from .binary import *

__all__ = [s for s in dir() if not s.startswith("_")]
