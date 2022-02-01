"""
====================================================
Polar plot utilities (:mod:`fklab.plot.plots.polar`)
====================================================

.. currentmodule:: fklab.plot.plots.polar

Function for polar plots.

"""
from ..polar import *

__all__ = [
    "polar_colorbar",
    "plot_polar_map",
    "set_polar_axes",
    "setup_polar_axes_grid",
    "float_radial_axis",
    "style_polar_axes",
    "annotate_angles",
]

import warnings
from fklab.codetools import FKLabDeprecationWarning

warnings.warn(
    "The fklab.plot.polts.polar module is deprecated. "
    "Please use fklab.plot.polar instead.",
    FKLabDeprecationWarning,
    stacklevel=2,
)
