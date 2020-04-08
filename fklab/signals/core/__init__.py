"""
===========================================
Core Algorithms (:mod:`fklab.signals.core`)
===========================================

.. currentmodule:: fklab.signals.core

Basic algorithms.


"""
from .basic_algorithms import compute_threshold
from .basic_algorithms import compute_threshold_median
from .basic_algorithms import compute_threshold_percentile
from .basic_algorithms import compute_threshold_zscore
from .basic_algorithms import detect_mountains
from .basic_algorithms import event_triggered_average
from .basic_algorithms import extract_data_windows
from .basic_algorithms import extract_trigger_windows
from .basic_algorithms import generate_trigger_windows
from .basic_algorithms import generate_windows
from .basic_algorithms import localextrema
from .basic_algorithms import localmaxima
from .basic_algorithms import localminima
from .basic_algorithms import remove_artefacts
from .basic_algorithms import zerocrossing

__all__ = [s for s in dir() if not s.startswith("_")]
