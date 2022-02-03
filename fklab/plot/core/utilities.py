"""
============================================
Utilities (:mod:`fklab.plot.core.utilities`)
============================================

.. currentmodule:: fklab.plot.core.utilities

Plotting utilities
"""
from ..utilities import ColumnView
from ..utilities import install_custom_colors
from ..utilities import install_custom_stylesheets
from ..utilities import LinearOffsetCollection
from ..utilities import RangeVector
from fklab.version._core_version._version import __version__

__all__ = [
    "LinearOffsetCollection",
    "RangeVector",
    "ColumnView",
    "install_custom_stylesheets",
    "install_custom_colors",
]

import warnings
from fklab.codetools import FKLabDeprecationWarning

warnings.warn(
    "The fklab.plot.core.utilities module is deprecated. "
    "Please use fklab.plot.utilities instead.",
    FKLabDeprecationWarning,
    stacklevel=2,
)
