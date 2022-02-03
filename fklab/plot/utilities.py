"""
============================================
Utilities (:mod:`fklab.plot.core.utilities`)
============================================

.. currentmodule:: fklab.plot.core.utilities

Plotting utilities
"""
import atexit as _atexit
import math
import pathlib as _pathlib

import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.style as _style
import matplotlib.transforms
import numpy as np
import pkg_resources as _pkg

from fklab.version._core_version._version import __version__


__all__ = [
    "LinearOffsetCollection",
    "RangeVector",
    "ColumnView",
    "install_custom_stylesheets",
    "install_custom_colors",
    "axes_annotation",
    "setup_axes_grid",
    "fixed_colorbar",
]


def install_custom_stylesheets(pkg):
    _stylesheets = _pkg.resource_filename(pkg, "stylesheets")

    _atexit.unregister(_pkg.cleanup_resources)  # make sure it is only registered once
    _atexit.register(_pkg.cleanup_resources)

    styles = _pathlib.Path(_stylesheets).glob("*.mplstyle")
    styles = tuple(x.stem for x in styles)

    if not _stylesheets in _style.core.USER_LIBRARY_PATHS:
        _style.core.USER_LIBRARY_PATHS.append(_stylesheets)
        _style.core.reload_library()

    return styles


def install_custom_colors(colors, name=None):
    # first validate colors
    colors = {k: mpl.colors.to_hex(v) for k, v in colors.items()}

    if not name is None:
        mpl.colors.__dict__["{}_COLORS".format(str(name).upper())] = colors

    mpl.colors.colorConverter.colors.update(colors)
    mpl.colors.colorConverter.cache.clear()


class LinearOffsetCollection:
    """Helper class to layout artists in axes.

    Parameters
    ----------
    children : list
        List of artists.
    spacing : float
        Offsets between children (can be negative).
    origin : float
        Base offset.
    direction : {'vertical', 'horizontal'}
        Apply offsets in horizontal or vertical direction.

    """

    def __init__(self, children=[], spacing=0.0, origin=0.0, direction="vertical"):
        self._children = children
        self._spacing = float(spacing)
        self._origin = float(origin)

        self._vertical = True if direction in ("vertical", "v", "vert") else False

        self._apply_offset()

    def set_origin(self, origin):
        self._origin = float(origin)
        self._apply_offset()

    def set_spacing(self, spacing):
        self._spacing = float(spacing)
        self._apply_offset()

    def set_vertical(self, val=True):
        self._vertical = bool(val)
        self._apply_offset()

    def set_horizontal(self, val=True):
        self._vertical = not bool(val)
        self._apply_offset()

    def set_direction(self, val):
        self._vertical = True if val in ("vertical", "v", "vert") else False
        self._apply_offset()

    def __len__(self):
        return self._children.__len__()

    def __getitem__(self, key):
        return self._children.__getitem__(key)

    def __setitem__(self, key, value):
        self._children.__setitem__(key, value)
        self._apply_offset()

    def __delitem__(self, key):
        self._children.__delitem__(key)
        self._apply_offset()

    def append(self, value):
        self._children.append(value)
        self._apply_offset()

    def extend(self, values):
        self._children.extend(values)
        self._apply_offset()

    def insert(self, index, value):
        self._children.insert(index, value)
        self._apply_offset()

    def sort(self, cmp=None, key=None, reverse=False):
        self._children.sort(cmp=cmp, key=key, reverse=reverse)
        self._apply_offset()

    def update(self):
        self._apply_offset()

    def _apply_offset(self):

        if self._vertical:
            translation = [0, self._origin]
            index = 1
        else:
            translation = [self._origin, 0]
            index = 0

        for child in self._children:
            t = mpl.transforms.Affine2D()
            t.translate(*translation)
            child.set_transform(t + child.axes.transData)
            translation[index] += self._spacing


class RangeVector:
    """Handle lazily computed range of values.

    Parameters
    ----------
    n : int
        Number of values.
    start : float, optional
        Start value.
    delta : float, optional
        Spacing between values.

    """

    def __init__(self, n, start=0, delta=1):
        self._n = int(n)
        self._start = float(start)
        self._delta = float(delta)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return np.arange(*key.indices(self._n)) * self._delta + self._start
        elif isinstance(key, int):
            if key < 0:
                key += self._n
            if key < 0 or key >= self._n:
                raise KeyError("Key out of bounds")
            return key * self._delta + self._start
        else:
            raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __len__(self):
        return self._n

    @property
    def shape(self):
        return [self._n]

    @property
    def ndim(self):
        return 1


class ColumnView(object):
    """Handle column view on a numpy 2d array.

    A ColumnView object represents a read-only column in a 2d numpy array.
    This class is needed for HDF5 data arrays that do not provide views,
    but rather load the data when indexed. Optionally, a function can be
    applied to the indexed data.

    Parameters
    ----------
    source : 2d array
    col : int, optional
        Column index.
    fcn : callable, optional
        Function that is called when data is requested. The function should
        work element-wise and should not change the shape of the array.

    """

    def __init__(self, source, col=0, fcn=None):

        if not source.ndim == 2:
            raise TypeError("Expecting 2-d array")

        self._source = source
        self._col = int(col)
        self._fcn = fcn

    def __len__(self):
        return len(self._source)

    @property
    def shape(self):
        return (self._source.shape[0],)

    @property
    def ndim(self):
        return 1

    def __getitem__(self, key):

        if self._fcn is None:
            return self._source[key, self._col]
        else:
            return self._fcn(self._source[key, self._col])


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
