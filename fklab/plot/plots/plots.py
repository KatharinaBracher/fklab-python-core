"""
==========================================================
General plotting functions (:mod:`fklab.plot.plots.plots`)
==========================================================

.. currentmodule:: fklab.plot.plots.plots

Function for plotting multiple signals, events, segments and rasters.

"""
from ..plotfunctions import *
from fklab.version._core_version._version import __version__

__all__ = [
    "enhanced_scatter",
    "plot_1d_maps",
    "plot_2d_maps",
    "plot_signals",
    "plot_events",
    "plot_segments",
    "plot_raster",
    "plot_event_image",
    "plot_event_raster",
    "plot_spectrogram",
    "add_static_scalebar",
    "add_scalebar",
    "add_axes_message",
    "labeled_vmarker",
    "labeled_hmarker",
]

import warnings
from fklab.codetools import FKLabDeprecationWarning

warnings.warn(
    "The fklab.plot.plots.plots module is deprecated. "
    "Please use fklab.plot.plotfunctions instead.",
    FKLabDeprecationWarning,
    stacklevel=2,
)
