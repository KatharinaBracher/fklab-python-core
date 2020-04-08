"""
==================================================
Plot utilities (:mod:`fklab.plot.plots.utilities`)
==================================================

.. currentmodule:: fklab.plot.plots.utilities

Utility plot functions.


"""
import math

import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["axes_annotation", "setup_axes_grid", "fixed_colorbar"]


def axes_annotation(ax, text, hloc="right", vloc="top out", **kwargs):
    """Add text annotation to axes.

    Parameters
    ----------
    ax : Axes object
    text : str
    hloc : str
        Horizontal location of text. Valid options are: "left", "right"
        "center", "left in", "left out", "right in" and "right out".
    vloc : str
        Vertical location of text. Valid options are: "top", "bottom"
        "center", "top in", "top out", "bottom in" and "bottom out".
    **kwargs
        Extra keyword arguments for *text* function.

    Returns
    -------
    Text

    """
    if "right" in hloc:
        x = 1
        ha = "left" if "out" in hloc else "right"
    elif "left" in hloc:
        x = 0
        ha = "right" if "out" in hloc else "left"
    elif hloc == "center":
        x = 0.5
        ha = "center"
    else:
        raise ValueError("Unrecognizable hloc argument.")

    if "top" in vloc:
        y = 1
        va = "bottom" if "out" in vloc else "top"
    elif "bottom" in vloc:
        y = 0
        va = "top" if "out" in vloc else "bottom"
    elif vloc == "center":
        y = 0.5
        va = "center"
    else:
        raise ValueError("Unrecognizable vloc argument.")

    return ax.text(x, y, text, va=va, ha=ha, transform=ax.transAxes, **kwargs)


def setup_axes_grid(
    n,
    grid=None,
    figsize=None,
    axsize=None,
    grid_kw={},
    sharex=False,
    sharey=False,
    subplot_kw={},
):
    """Construct figure with grid of axes.

    Parameters
    ----------
    n : int
    grid : None or int or (int, int) or (None, int) or (int, None)
    figsize : (width, height)
    axsize : (width, height)
    grid_kw : dict
    sharex, sharey : bool
    subplot_kw : dict

    Returns
    -------
    fig : Figure
    ax : array of Axes

    """
    if grid is None:
        grid = (None, None)
    elif isinstance(grid, int):
        grid = (None, grid)

    nrows, ncols = grid

    if nrows is None and ncols is None:
        ncols = math.ceil(math.sqrt(n))
        nrows = (n + (ncols - 1)) // ncols
    elif nrows is None:
        nrows = (n + (ncols - 1)) // ncols
    else:
        ncols = (n + (nrows - 1)) // nrows

    if not axsize is None:
        if not figsize is None:
            raise ValueError("Provide figsize or axsize, but not both.")
        figsize = (axsize[0] * ncols, axsize[1] * nrows)

    fig, ax = plt.subplots(
        nrows,
        ncols,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        subplot_kw=subplot_kw,
        gridspec_kw=grid_kw,
        squeeze=False,
    )

    for a in ax.ravel()[n:]:
        a.axis("off")

    return fig, ax


def fixed_colorbar(vmin=0, vmax=1, cmap="inferno", ndivisions=5, **kwargs):
    """Colorbar with fixed min/max values.

    Parameters
    ----------
    vmin, vmax : float or str
    cmap : colormap
    ndivisions : int
    **kwargs
        Extra keyword arguments for *plt.colorbar* function.

    Returns
    -------
    colorbar

    """
    mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
    cbar = plt.colorbar(mappable, **kwargs)
    cbar.set_ticks(np.linspace(0, 1, ndivisions + 1))
    cbar.set_ticklabels([vmin] + [""] * (ndivisions - 1) + [vmax])
    return cbar
