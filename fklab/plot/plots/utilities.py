"""
==================================================
Plot utilities (:mod:`fklab.plot.plots.utilities`)
==================================================

.. currentmodule:: fklab.plot.plots.utilities

Utility plot functions.


"""
from ..utilities import axes_annotation
from ..utilities import fixed_colorbar
from ..utilities import setup_axes_grid

__all__ = ["axes_annotation", "setup_axes_grid", "fixed_colorbar"]

import warnings
from fklab.codetools import FKLabDeprecationWarning

warnings.warn(
    "The fklab.plot.plots.utilities module is deprecated. "
    "Please use fklab.plot.utilities instead.",
    FKLabDeprecationWarning,
    stacklevel=2,
)
