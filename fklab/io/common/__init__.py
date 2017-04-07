"""
=================================================
Common File IO Utilities (:mod:`fklab.io.common`)
=================================================

.. currentmodule:: fklab.io.common

Common file import and export utilities.

.. automodule:: fklab.io.common.binary
    
"""

from .binary import *

__all__ = [s for s in dir() if not s.startswith('_')]
