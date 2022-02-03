"""
================================================
Interaction (:mod:`fklab.plot.core.interaction`)
================================================

.. currentmodule:: fklab.plot.core.interaction

Utilities for interaction with matplotlib figures
"""
from ..interaction import *
from fklab.version._core_version._version import __version__

__all__ = [
    "interactive_figure",
    "iEllipse",
    "iRectangle",
    "iPolyline",
    "iPolygon",
    "create_square",
    "create_rectangle",
    "create_circle",
    "create_ellipse",
    "create_polyline",
    "create_polygon",
    "ScrollPanZoom",
]

import warnings
from fklab.codetools import FKLabDeprecationWarning

warnings.warn(
    "The fklab.plot.core.interaction module is deprecated. "
    "Please use fklab.plot.interaction instead.",
    FKLabDeprecationWarning,
    stacklevel=2,
)
