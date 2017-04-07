"""
=================================================
Rate conversions (:mod:`fklab.signals.multirate`)
=================================================

.. currentmodule:: fklab.signals.multirate

Sampling rate conversion functions.

.. automodule:: fklab.signals.multirate.multirate
    
"""

from .multirate import *

__all__ = [s for s in dir() if not s.startswith('_')]
