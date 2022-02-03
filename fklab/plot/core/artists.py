"""
=================================================================
Artists (:mod:`fklab.plot.core.artists`)
=================================================================

.. currentmodule:: fklab.plot.core.artists

Custom matplotlib artists

"""
from ..artists import *
from fklab.version._core_version._version import __version__


__all__ = [
    "EventImage",
    "FastLine",
    "FastSpectrogram",
    "StaticScaleBar",
    "AnchoredScaleBar",
    "AxesMessage",
    "PositionTimeStrip",
]

import warnings
from fklab.codetools import FKLabDeprecationWarning

warnings.warn(
    "The fklab.plot.core.artists module is deprecated. "
    "Please use fklab.plot.artists instead.",
    FKLabDeprecationWarning,
    stacklevel=2,
)
