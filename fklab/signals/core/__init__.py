"""
===========================================
Core Algorithms (:mod:`fklab.signals.core`)
===========================================

.. currentmodule:: fklab.signals.core

Basic algorithms.

.. automodule:: fklab.signals.core.basic_algorithms
    
"""

from .basic_algorithms import (detect_mountains, zerocrossing, localmaxima, 
    localminima, localextrema, remove_artefacts, extract_data_windows,
    extract_trigger_windows, generate_windows, generate_trigger_windows,
    event_triggered_average)

__all__ = [s for s in dir() if not s.startswith('_')]
