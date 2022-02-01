"""
==========================================================
General plotting functions (:mod:`fklab.plot.plots.plots`)
==========================================================

.. currentmodule:: fklab.plot.plots.plots

Function for plotting multiple signals, events, segments and rasters.

"""
import itertools

import matplotlib.cm
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

import fklab.codetools
import fklab.segments
from .artists import AnchoredScaleBar
from .artists import AxesMessage
from .artists import EventImage
from .artists import FastLine
from .artists import FastSpectrogram
from .artists import StaticScaleBar
from .interaction import ScrollPanZoom
from .utilities import ColumnView
from .utilities import fixed_colorbar
from .utilities import LinearOffsetCollection
from .utilities import RangeVector
from .utilities import setup_axes_grid
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


def enhanced_scatter(*args, rotations=None, alphas=None, axes=None, **kwargs):
    """Scatter plot that supports varying rotation and alpha for each marker.

    Parameters
    ----------
    \*args, \*\*kwargs :
        Arguments for the matplotlib `scatter` function
    rotations : scalar or 1d array-like
        Marker rotations.
    alphas : scalar or 1d array-like
        Marker alpha values

    Returns
    -------
    PathCollection

    """
    if axes is None:
        axes = plt.gca()

    h = axes.scatter(*args, **kwargs)

    if not rotations is None:
        rotations = np.ravel(rotations)

        paths = h.get_paths()

        if len(paths) == 1:
            paths = [
                p.transformed(matplotlib.transforms.Affine2D().rotate(phi))
                for p, phi in zip(itertools.cycle(paths), rotations)
            ]
        else:
            paths = [
                p.transformed(matplotlib.transforms.Affine2D().rotate(phi))
                for p, phi in zip(paths, itertools.cycle(rotations))
            ]

        h.set_paths(paths)

    if not alphas is None:
        alphas = np.ravel(alphas)

        fc = h.get_facecolors()

        if len(fc) > 0:
            alphas_ = alphas
            if len(fc) == 1:
                fc = np.broadcast_to(fc, (len(alphas), 4)).copy()
            elif len(alphas) == 1:
                alphas_ = np.broadcast_to(alphas, len(fc))

            fc[:, 3] *= alphas_

            h.set_facecolors(fc)

        ec = h.get_edgecolors()

        if len(ec) > 0:
            alphas_ = alphas
            if len(ec) == 1:
                ec = np.broadcast_to(ec, (len(alphas), 4)).copy()
            elif len(alphas) == 1:
                alphas_ = np.broadcast_to(alphas, len(ec))

            ec[:, 3] *= alphas_

            h.set_facecolors(ec)

    return h


def plot_1d_maps(
    maps,
    x=None,
    xlabel="",
    ylabel="",
    color="k",
    fill=True,
    fill_alpha=0.5,
    emin=0,
    emax=0.1,
    vmin="shared",
    vmax="shared",
    roundto=None,
    **kwargs
):
    """Plot grid of 1D arrays.

    Parameters
    ----------
    maps : iterable of 1d arrays
    x : (start,end) or 1d array-like
        X coordinates
    color : color spec
    fill : bool
    fill_alpha : float
    xlabel, ylabel : str
    vmin, vmax : 'shared', 'auto', scalar
        The minimum and maximum y value. If 'shared', the value will be
        computed as the min/max across all *maps*. If 'auto', the value will
        be computed separately for each map and will be displayed above each
        plot (minimum on the top left, maximum on the top right). If a scalar,
        the value will be fixed for all plots.
    roundto : scalar
        Round down/up the minimum and maximum y values to the nearest
        multiple of *roundto*.
    emin, emax : scalar
        Extra space in the y dimensions below/above the data.
    \*\*kwargs :
        Extra keyword arguments for *setup_axes_grid* function.

    Returns
    -------
    fig : Figure
    axes : (nrows, ncols) array of Axes

    """
    nmaps = len(maps)

    fig, axes = setup_axes_grid(nmaps, **kwargs)

    npoints = len(maps[0])

    if not all([len(m) == npoints for m in maps]):
        raise ValueError()

    if x is None:
        x = np.arange(npoints)
    elif len(x) == 2:
        x = np.linspace(*x, npoints)
    elif len(x) == npoints + 1:
        # bins, convert to centers
        x = (x[:-1] + x[1:]) / 2
    elif len(x) != npoints:
        raise ValueError()

    if vmin == "shared":
        vmin_value = [np.nanmin([np.nanmin(m[np.isfinite(m)]) for m in maps])] * nmaps
    elif vmin == "auto":
        vmin_value = [np.nanmin(m[np.isfinit(m)]) for m in maps]
    elif isinstance(vmin, (int, float)):
        vmin_value = [vmin] * nmaps
    else:
        raise ValueError()

    if vmax == "shared":
        vmax_value = [np.nanmax([np.nanmax(m[np.isfinite(m)]) for m in maps])] * nmaps
    elif vmax == "auto":
        vmax_value = [np.nanmax(m[np.isfinite(m)]) for m in maps]
    elif isinstance(vmax, (int, float)):
        vmax_value = [vmax] * nmaps
    else:
        raise ValueError()

    vmin_value = np.array(vmin_value)
    vmax_value = np.array(vmax_value)

    if emin + emax >= 1:
        raise ValueError()

    delta = vmax_value - vmin_value

    if vmin == "auto":
        vmin_value = vmin_value - (emin * delta / (1 - emin - emax))
    if vmax == "auto":
        vmax_value = vmax_value + (emax * delta / (1 - emin - emax))

    if not roundto is None:
        vmin_value = np.floor(vmin_value / roundto) * roundto
        vmax_value = np.ceil(vmax_value / roundto) * roundto

    axes = np.atleast_2d(axes)

    for data, ax, minval, maxval in zip(maps, axes.ravel(), vmin_value, vmax_value):

        if fill:
            ax.fill_between(x, data, color=color, alpha=fill_alpha, lw=0)

        ax.plot(x, data, color=color)

        ax.set_ylim((minval, maxval))
        ax.set_xlim((x[0], x[-1]))

        if vmin == "auto":
            ax.text(x[0], maxval, "{}".format(minval), va="bottom", ha="left")
        if vmax == "auto":
            ax.text(x[-1], maxval, "{}".format(maxval), va="bottom", ha="right")

        if not ax is axes[-1, 0]:
            ax.set(yticks=[], xticks=[])
        else:
            ax.set(ylabel=ylabel, xlabel=xlabel)
            ax.set(
                yticks=np.linspace(minval, maxval, 6),
                yticklabels=["min" if vmin == "auto" else minval]
                + [""] * 4
                + ["max" if vmax == "auto" else maxval],
                xticks=[x[0], x[-1]],
            )

    return fig, axes


def plot_2d_maps(
    maps,
    coordinates=None,
    xlabel="",
    ylabel="",
    colorlabel="",
    cmap="inferno",
    cbar=True,
    cbar_kw={},
    cmin="shared",
    cmax="shared",
    roundto=None,
    **kwargs
):
    """Plot image grid of 2D arrays.

    Parameters
    ----------
    maps : iterable of 2d arrays
    coordinates : [(xmin, xmax),(ymin,ymax)] or [array, array]
        Coordinates along the x and y dimensions. For each dimension,
        coordinates can either be specified as (min,max) tuple, a (n+1,) array
        with bin edges or a (n,) array with coordinates for all *n*
        rows/columns in the map.
    cmap : colormap
    xlabel, ylabel, colorlabel : str
        Labels for x and y axes and for the color bar
    cbar : bool
        Show color bar.
    cbar_kw: {}
        Extra keyword arguments for colorbar.
    cmin, cmax : 'shared', 'auto', scalar
        The minimum and maximum color value. If 'shared', the value will be
        computed as the min/max across all *maps*. If 'auto', the value will
        be computed separately for each map and will be displayed above each
        plot (minimum on the top left, maximum on the top right). If a scalar,
        the value will be fixed for all plots.
    roundto : scalar
        Round down/up the minimum and maximum color values to the nearest
        multiple of *roundto*.
    \*\*kwargs :
        Extra keyword arguments for *setup_axes_grid* function.

    Returns
    -------
    fig : Figure
    axes : (nrows, ncols) array of Axes

    """
    nmaps = len(maps)

    if nmaps < 1:
        raise ValueError("Expecting at least one 2d map.")

    fig, axes = setup_axes_grid(nmaps, **kwargs)

    maps = [np.asarray(m) for m in maps]

    shape = maps[0].shape
    if len(shape) != 2 or any([m.shape != shape for m in maps]):
        raise ValueError("All maps should have the same 2d shape.")

    if coordinates is None:
        extent = None
    else:
        extent = np.zeros(4)
        for c, slc, nn in zip(coordinates, [slice(0, 2), slice(2, None)], shape):
            if len(c) == 2:
                extent[slc] = c
            elif len(c) == nn:
                cdiff = np.diff(c)
                extent[slc] = [c[0] - cdiff[0] / 2, c[-1] + cdiff[-1] / 2]
            elif len(c) == nn + 1:  # bins
                extent[slc] = [c[0], c[-1]]
            else:
                raise ValueError("Incorrect size of coordinates arrays.")

    if cmin == "shared":
        cmin_value = [np.nanmin([np.nanmin(m[np.isfinite(m)]) for m in maps])] * nmaps
    elif cmin == "auto":
        cmin_value = [np.nanmin(m[np.isfinite(m)]) for m in maps]
    elif isinstance(cmin, (int, float)):
        cmin_value = [cmin] * nmaps
    else:
        raise ValueError()

    if cmax == "shared":
        cmax_value = [np.nanmax([np.nanmax(m[np.isfinite(m)]) for m in maps])] * nmaps
    elif cmax == "auto":
        cmax_value = [np.nanmax(m[np.isfinite(m)]) for m in maps]
    elif isinstance(cmax, (int, float)):
        cmax_value = [cmax] * nmaps
    else:
        raise ValueError()

    if not roundto is None:
        cmin_value = np.floor(np.array(cmin_value) / roundto) * roundto
        cmax_value = np.ceil(np.array(cmax_value) / roundto) * roundto

    cmin_label = "min" if cmin == "auto" else cmin_value[0]
    cmax_label = "max" if cmax == "auto" else cmax_value[0]

    axes = np.atleast_2d(axes)

    for data, ax, minval, maxval in zip(maps, axes.ravel(), cmin_value, cmax_value):

        img = ax.imshow(
            data.T,
            interpolation="none",
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
        )

        img.set_clim((minval, maxval))

        if cmin == "auto":
            ax.text(extent[0], extent[3], "{}".format(minval), va="bottom", ha="left")
        if cmax == "auto":
            ax.text(extent[1], extent[3], "{}".format(maxval), va="bottom", ha="right")

        if not ax is axes[-1, 0]:
            ax.set(yticks=[], xticks=[])
        else:
            ax.set(ylabel=ylabel, xlabel=xlabel)

    if cbar:
        cax = fixed_colorbar(
            vmin=cmin_label, vmax=cmax_label, cmap=cmap, ax=axes, **cbar_kw
        )
        cax.set_label(colorlabel)

    return fig, axes


def plot_signals(*args, **kwargs):
    """Plot multiple time series.

    Parameters
    ----------
    x : 1D array-like or list of 1D-array like, optional
    y : array-like or list of array like
    spacing : scalar , optional
    origin : scalar, optional
    lineclass : {Line2D, FastLine}, optional
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : line properties

    Returns
    -------
    collection of lines

    """
    if len(args) < 1 or len(args) > 2:
        raise ValueError("Expecting one or two positional data arguments")

    if len(args) == 1:
        x = None
        y, = args
    else:
        x, y = args

    # TODO: check x and y inputs
    # how many arrays? how many signals? how many samples?
    if not isinstance(y, list):
        y = [y]

    if not all([len(p.shape) == 1 or len(p.shape) == 2 for p in y]):
        raise ValueError("Invalid data")

    if not isinstance(x, list):
        x = [x] * len(y)
    elif len(x) != len(y):
        raise ValueError("Invalid time vectors")

    if not all([p is None or len(p.shape) == 1 for p in x]):
        raise ValueError("Invalid time vectors")

    data_shapes = np.array(
        [[p.shape[0], 1 if len(p.shape) == 1 else p.shape[1]] for p in y]
    )
    time_shapes = np.array([-1 if p is None else p.shape[0] for p in x])

    if np.any(np.logical_and(data_shapes[:, 0] != time_shapes, time_shapes != -1)):
        raise ValueError("Invalid data")

    nsignals = np.sum(data_shapes[:, 1])

    # get LinearOffsetCollection keywords
    spacing = kwargs.pop("spacing", 0.0)
    origin = kwargs.pop("origin", 0.0)

    cm = kwargs.pop("colormap", None)

    if cm is not None:
        alpha = kwargs.pop("alpha", 1.0)
        if isinstance(cm, str):
            cm = matplotlib.cm.get_cmap(cm, lut=nsignals)

    color = kwargs.pop("color", None)

    if cm is not None and color is not None:
        raise ValueError("Specify either color or colormap, but not both")

    lineclass = kwargs.pop("lineclass", FastLine)

    lines = []

    ax = kwargs.pop("axes", None)

    if ax is None:
        ax = plt.gca()

    def local_color(k):
        if cm is not None:
            return cm(k % cm.N, alpha=alpha)
        else:
            return color

    # create lines
    signal_index = 0
    for xdata, ydata in zip(x, y):

        # if cm is not None:
        #    color = cm(signal_index % cm.N, alpha=alpha)
        #    signal_index += 1

        if xdata is None:
            if ydata.ndim == 1:
                lines.append(
                    lineclass(
                        RangeVector(len(ydata)),
                        ydata,
                        color=local_color(signal_index),
                        **kwargs
                    )
                )
                signal_index += 1
            else:
                for idx in range(ydata.shape[1]):
                    lines.append(
                        lineclass(
                            RangeVector(ydata.shape[0]),
                            ColumnView(ydata, idx),
                            color=local_color(signal_index),
                            **kwargs
                        )
                    )
                    signal_index += 1
        else:
            if ydata.ndim == 1:
                lines.append(
                    lineclass(xdata, ydata, color=local_color(signal_index), **kwargs)
                )
                signal_index += 1
            else:
                for idx in range(ydata.shape[1]):
                    lines.append(
                        lineclass(
                            xdata,
                            ColumnView(ydata, idx),
                            color=local_color(signal_index),
                            **kwargs
                        )
                    )
                    signal_index += 1

    for l in lines:
        ax.add_line(l)

    collection = LinearOffsetCollection(
        lines, spacing=spacing, origin=origin, direction="vertical"
    )

    plt.draw()

    return collection


@fklab.codetools.deprecated("Please use matplotlib's eventplot or plot_event_raster")
def plot_events(events, **kwargs):
    """Plot multiple event series.

    Parameters
    ----------
    events : 1D array-like or list of 1D-array like
    kind : {'event', 'raster'}, optional
    spacing : scalar , optional
    origin : scalar, optional
    fullheight : bool, optional
    axes : Axes object
    colormap : str or matplotlib colormap object
    \*\*kwargs : EventCollection or EventImage options

    Returns
    -------
    collection of events

    """
    if not isinstance(events, list):
        events = [events]

    if not all([p.ndim == 1 for p in events]):
        raise ValueError("Invalid data")

    kind = kwargs.pop("kind", "event")

    # get LinearOffsetCollection keywords
    spacing = kwargs.pop("spacing", 0.0)
    origin = kwargs.pop("origin", 0.0)

    fullheight = kwargs.pop("fullheight", False)

    if fullheight:
        kwargs.update(lineoffset=0.5)

    cm = kwargs.pop("colormap", None)

    alpha = kwargs.pop("alpha", 1.0)

    if cm is not None:
        if isinstance(cm, str):
            cm = matplotlib.cm.get_cmap(cm, lut=len(events))

    color = kwargs.pop("color", None)

    if cm is not None and color is not None:
        raise ValueError("Specify either color or colormap, but not both")

    ax = kwargs.pop("axes", None)

    if ax is None:
        ax = plt.gca()

    items = []

    if kind == "event":
        for k in range(len(events)):
            if cm is not None:
                color = cm(k % cm.N, alpha=alpha)
            items.append(
                matplotlib.collections.EventCollection(events[k], color=color, **kwargs)
            )
    elif kind == "raster":
        for k in range(len(events)):
            if cm is not None:
                color = cm(k % cm.N)
            items.append(
                EventImage(
                    ax,
                    events[k],
                    foreground_color=color,
                    foreground_alpha=alpha,
                    **kwargs
                )
            )
    else:
        raise ValueError("Unknown value for parameter kind")

    for item in items:
        if kind == "event":
            ax.add_collection(item)
        else:
            ax.add_image(item)

        if fullheight:
            item.set_transform(
                matplotlib.transforms.blended_transform_factory(
                    ax.transData, ax.transAxes
                )
            )

    if not fullheight:
        items = LinearOffsetCollection(
            items, spacing=spacing, origin=origin, direction="vertical"
        )

    plt.draw()

    return items


@fklab.codetools.deprecated("Please use the plotting functions in fklab.segments")
def plot_segments(s, **kwargs):
    """Plot multiple segment series.

    Parameters
    ----------
    segments : Nx2 array-like or list of Nx2 array-like
    spacing : scalar , optional
    origin : scalar, optional
    fullheight : bool, optional
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : BrokenBarHCollection options

    Returns
    -------
    collection of BrokenBarHCollection

    """
    if not isinstance(s, list):
        s = [s]

    s = [fklab.segments.check_segments(k, copy=True) for k in s]

    # get LinearOffsetCollection keywords
    spacing = kwargs.pop("spacing", 0.0)
    origin = kwargs.pop("origin", 0.0)

    fullheight = kwargs.pop("fullheight", False)

    cm = kwargs.pop("colormap", None)

    if cm is not None:
        alpha = kwargs.pop("alpha", 1.0)
        if isinstance(cm, str):
            cm = matplotlib.cm.get_cmap(cm, lut=len(s))

    color = kwargs.pop("color", None)

    if cm is not None and color is not None:
        raise ValueError("Specify either color or colormap, but not both")

    ax = kwargs.pop("axes", None)

    if ax is None:
        ax = plt.gca()

    items = []

    for k in range(len(s)):
        if cm is not None:
            color = cm(k % cm.N, alpha=alpha)
        s[k][:, 1] = s[k][:, 1] - s[k][:, 0]
        items.append(
            matplotlib.collections.BrokenBarHCollection(
                s[k], [0, 1], facecolors=color, **kwargs
            )
        )

    for item in items:
        ax.add_collection(item)

    if fullheight:
        for item in items:
            item.set_transform(
                matplotlib.transforms.blended_transform_factory(
                    ax.transData, ax.transAxes
                )
            )
    else:
        items = LinearOffsetCollection(
            items, spacing=spacing, origin=origin, direction="vertical"
        )

    plt.draw()

    return items


@fklab.codetools.deprecated("Please use matplotlib's eventplot or plot_event_raster")
def plot_raster(events, **kwargs):
    """Plot multiple event series.

    Parameters
    ----------
    events : 1D array-like or list of 1D-array like
    spacing : scalar , optional
    origin : scalar, optional
    fullheight : bool, optional
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : EventImage options

    Returns
    -------
    collection of rasters

    """
    kwargs["kind"] = "raster"
    return plot_events(events, **kwargs)


def plot_event_image(events, axes=None, **kwargs):
    """Plot events after conversion to image.

    By converting visible events in the axes to an image, the drawing
    can be much faster than plotting a line for each event in case
    of a large number events and/or a large number of event series.
    This is especially useful for interactive exploration of the data.

    Parameters
    ----------
    events : 1d array-like
    axes : None or matplotlob Axes
        Destination axes. Will use `plt.gca()` is no axes is specified.
    offset : float
        Offset along y-axis. Event markers will be centered on `y`.
    height : float
        Height of the event markers.
    linewidth : int
        Width of event markers in pixels.
    foreground_color, background_color : color specification
        Color of the event markers (foregrond) and the background
    foreground_alpha, background_alpha : float
        Transparency for the event markers (foreground) and background

    Returns
    -------
    EventImage

    """

    if axes is None:
        axes = plt.gca()

    img = EventImage(axes, events, **kwargs)

    if img.get_clip_path() is None:
        # image does not already have clipping set, clip to axes patch
        img.set_clip_path(axes.patch)

    img.set_extent(img.get_extent())

    axes.add_image(img)

    return img


def plot_event_raster(
    events,
    axes=None,
    offsets=None,
    heights=1,
    linewidths=1,
    colors="black",
    alphas=None,
    background_colors="white",
    background_alphas=0,
    labels=None,
    label_colors=None,
):
    """Plot raster of multiple event series.

    Parameters
    ----------
    events : sequence of 1d array-like
    axes : None or matplotlob Axes
        Destination axes. Will use `plt.gca()` is no axes is specified.
    offsets : float or sequence of floats
        Offset along the y-axis for each event series.
    heights : float or sequence of floats
        Height of the event markers for each event series.
    linewidths : int of sequence of ints
        Width of event markers in pixels for each event series.
    colors, background_colors : color specification or sequence of colors
        Color of the event markers and background for each event series.
    alphas, background_alphas : float or sequence of floats
        Transparency for the event markers and background for each event series.
    labels : sequence of str
        Label for each event series.
    label_colors : color specification or sequence of colors
        Color for each label.

    Returns
    -------
    list of EventImage objects

    """

    if axes is None:
        axes = plt.gca()

    n = len(events)

    if offsets is None:
        offsets = np.arange(n, 0, -1)
    elif isinstance(offsets, (int, float)):
        offsets = np.full(n, offsets)

    if heights is None:
        heights = np.ones(n)
    elif isinstance(heights, (int, float)):
        heights = np.full(n, heights)

    if linewidths is None:
        linewidths = np.ones(n)
    elif isinstance(linewidths, (int, float)):
        linewidths = np.full(n, linewidths)

    if not isinstance(colors, (tuple, list)):
        colors = [colors] * n

    if alphas is None:
        alphas = np.ones(n)
    elif isinstance(alphas, (int, float)):
        alphas = np.full(n, alphas)

    if not isinstance(background_colors, (tuple, list)):
        background_colors = [background_colors] * n

    if background_alphas is None:
        background_alphas = np.ones(n)
    elif isinstance(background_alphas, (int, float)):
        background_alphas = np.full(n, background_alphas)

    if not label_colors is None and not isinstance(label_colors, (tuple, list)):
        label_colors = [label_colors] * n

    img = [
        plot_event_image(
            event,
            axes=axes,
            y=offsets[k],
            height=heights[k],
            linewidth=linewidths[k],
            foreground_color=colors[k],
            foreground_alpha=alphas[k],
            background_color=background_colors[k],
            background_alpha=background_alphas[k],
        )
        for k, event in enumerate(events)
    ]

    if not labels is None:
        axes.set(yticks=offsets, yticklabels=labels)

    if not labels is None and not label_colors is None:
        for ticklabel, tickcolor in zip(axes.get_yticklabels(), label_colors):
            ticklabel.set_color(tickcolor)

    amin = np.argmin(offsets)
    amax = np.argmax(offsets)

    axes.set(ylim=(offsets[amin] - heights[amin], offsets[amax] + heights[amax]))

    return img


def plot_spectrogram(signal, **kwargs):
    """Plot spectrogram of signal.

    Parameters
    ----------
    signal : 1D array-like
    axes : Axes object
    colormap : str or matplotlib colormap object
    kwargs : FastSpectrogram options

    Returns
    -------
    spectrogram image

    """
    if not signal.ndim == 1:
        raise ValueError("Invalid signal vector")

    ax = kwargs.pop("axes", None)

    if ax is None:
        ax = plt.gca()

    item = FastSpectrogram(signal, **kwargs)
    ax.add_image(item)
    plt.sci(item)

    plt.draw()

    return item


def add_static_scalebar(ax, hidex=True, hidey=True, **kwargs):
    """Add static scalebar to axes.

    Parameters
    ----------
    ax : axis
    **kwargs : additional arguments passed to StaticScaleBar

    Returns
    -------
    StaticScaleBar

    """
    sb = StaticScaleBar(**kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)

    return sb


def add_axes_message(ax, **kwargs):
    """Add message artist to axes.

    Parameters
    ----------
    ax : Axes
    **kwargs : additional arguments passed to AxesMessage

    Returns
    -------
    AxesMessage

    """
    item = AxesMessage(**kwargs)
    ax.add_artist(item)

    return item


def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """Add scalebars to axes.

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    Parameters
    ----------
    ax : the axis to attach ticks to
    matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    hidex,hidey : if True, hide x-axis and y-axis of parent
    \*\*kwargs : additional arguments passed to AnchoredScaleBars

    Returns
    -------
    AnchoredScaleBar

    """

    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if matchx:
        kwargs["sizex"] = f(ax.xaxis)
        kwargs["labelx"] = str(kwargs["sizex"])
    if matchy:
        kwargs["sizey"] = f(ax.yaxis)
        kwargs["labely"] = str(kwargs["sizey"])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)

    return sb


def labeled_vmarker(
    x,
    text=None,
    text_x="right",
    text_y=0.01,
    text_offset=0,
    style=None,
    text_style=None,
    ax=None,
):

    if ax is None:
        ax = plt.gca()

    artists = {}

    style_ = {"linewidth": 1, "color": "k", "linestyle": "--"}
    if not style is None:
        style_.update(style)

    artists["line"] = ax.axvline(x, **style_)

    if not text is None:

        if text_x == "left":
            ha = "right"
            text_x = x - text_offset
        else:
            ha = "left"
            text_x = x + text_offset

        if text_y == "bottom":
            text_y = 0
        elif text_y == "top":
            text_y = 1
        elif text_y == "center":
            text_y = 0.5

        va = ["bottom", "center", "top"][round(text_y * 2)]

        tform = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes
        )

        text_style_ = {"color": "k", "rotation": 90, "va": va, "ha": ha}
        if not text_style is None:
            text_style_.update(text_style)

        artists["text"] = ax.text(text_x, text_y, text, transform=tform, **text_style_)

    return artists


def labeled_hmarker(
    y,
    text=None,
    text_x=0.99,
    text_y="top",
    text_offset=0,
    style=None,
    text_style=None,
    ax=None,
):

    if ax is None:
        ax = plt.gca()

    artists = {}

    style_ = {"linewidth": 1, "color": "k", "linestyle": "--"}
    if not style is None:
        style_.update(style)

    artists["line"] = ax.axhline(y, **style_)

    if not text is None:

        if text_y == "bottom":
            va = "top"
            text_y = y - text_offset
        else:
            va = "bottom"
            text_y = y + text_offset

        if text_x == "left":
            text_x = 0
        elif text_x == "right":
            text_x = 1
        elif text_x == "center":
            text_x = 0.5

        ha = ["left", "center", "right"][round(text_x * 2)]

        tform = matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )

        text_style_ = {"color": "k", "va": va, "ha": ha}
        if not text_style is None:
            text_style_.update(text_style)

        artists["text"] = ax.text(text_x, text_y, text, transform=tform, **text_style_)

    return artists


# WIP
def signals_plot(*args, **kwargs):

    labels = kwargs.pop("labels", None)
    interaction = kwargs.pop("interaction", True)
    xlabel = kwargs.pop("xlabel", "")
    ylabel = kwargs.pop("ylabel", "")

    ax = kwargs.pop("axes", None)
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    p = plot_signals(*args, axes=ax, **kwargs)

    ax.set_xlabel(xlabel)
    yl = ax.set_ylabel(ylabel)

    ax.set_yticks(np.array([-1, 0, 1]) * p._spacing[1])
    ax.spines["left"].set_bounds(-p._spacing[1], p._spacing[1])

    yl.set_transform(
        matplotlib.transforms.blended_transform_factory(
            matplotlib.transforms.IdentityTransform(), ax.transData
        )
    )
    yl.set_y(0.0)

    t = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)

    labels = ["a"] * len(p)

    for idx in range(len(p)):
        ax.text(
            1,
            idx * p._spacing[1],
            labels[idx],
            transform=t + matplotlib.transforms.Affine2D().translate(10, 0),
        )

    ax.set_ylim(-p._spacing[1], len(p) * p._spacing[1])
    plt.draw()

    if interaction:
        ax.scrollpanzoom = ScrollPanZoom(ax)

    # draw_scale_annotation

    return


def events_plot():
    # plot events with offset
    # set up ylim, ytick, yticklabels
    # set up ylabel, xlabel
    # set up zoom/pan
    pass


def segments_plot():
    pass


def raster_plot():
    pass
