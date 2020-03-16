"""
==========================================================
General plotting functions (:mod:`fklab.plot.plots.plots`)
==========================================================

.. currentmodule:: fklab.plot.plots.plots

Function for plotting multiple signals, events, segments and rasters.

plotting functions
==================

.. autosummary::
    :toctree: generated/

    plot_signals
    plot_events
    plot_segments
    plot_raster
    plot_spectrogram
    plot_1d_maps
    plot_2d_maps
    add_scalebar
    add_static_scalebar
    add_axes_message

    labeled_vmarker
    labeled_hmarker


"""
import matplotlib.cm
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

import fklab.segments
from fklab.plot.core.artists import AnchoredScaleBar
from fklab.plot.core.artists import AxesMessage
from fklab.plot.core.artists import FastLine
from fklab.plot.core.artists import FastRaster
from fklab.plot.core.artists import FastSpectrogram
from fklab.plot.core.artists import StaticScaleBar
from fklab.plot.core.interaction import ScrollPanZoom
from fklab.plot.core.utilities import ColumnView
from fklab.plot.core.utilities import LinearOffsetCollection
from fklab.plot.core.utilities import RangeVector
from fklab.version._core_version._version import __version__

__all__ = [
    "plot_1d_maps",
    "plot_2d_maps",
    "plot_signals",
    "plot_events",
    "plot_segments",
    "plot_raster",
    "plot_spectrogram",
    "add_static_scalebar",
    "add_scalebar",
    "add_axes_message",
    "labeled_vmarker",
    "labeled_hmarker",
]


def plot_1d_maps(maps, x=None, xlabel='', ylabel='', color='k', fill=True,
                 grid=None, figsize=None, fill_alpha=0.5, emin=0, emax=0.1,
                 vmin='shared', vmax='shared', roundto=None):
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
    grid : (rows,cols)
        Number of row and columns for the plot grid. Entries can be None,
        which means that the value will be automatically computed.
    figsize : (width, height)
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
    
    Returns
    -------
    fig : Figure
    ax : (nrows, ncols) array of Axes
    
    """
    
    nmaps = len(maps)
    npoints = len(maps[0])
    
    if not all([len(m)==npoints for m in maps]):
        raise ValueError()
    
    if grid is None:
        grid = (None, None)
    elif isinstance(grid, int):
        grid = (None, grid)
    
    nrows, ncols = grid
    
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(nmaps)))
        nrows = (nmaps + (ncols-1))//ncols
    elif nrows is None:
        nrows = (nmaps + (ncols-1))//ncols
    else:
        ncols = (nmaps + (nrows-1))//nrows
    
    if x is None:
        x = np.arange(npoints)
    elif len(x)==2:
        x = np.linspace(*x, npoints)
    elif len(x)!=npoints:
        raise ValueError()
    
    if vmin=='shared':
        vmin_value = [np.nanmin([np.nanmin(m[np.isfinite(m)]) for m in maps]),]*nmaps
    elif vmin=='auto':
        vmin_value = [np.nanmin(m[np.isfinit(m)]) for m in maps]
    elif isinstance(vmin, (int, float)):
        vmin_value = [vmin,]*nmaps
    else:
        raise ValueError()
        
    if vmax=='shared':
        vmax_value = [np.nanmax([np.nanmax(m[np.isfinite(m)]) for m in maps]),]*nmaps
    elif vmax=='auto':
        vmax_value = [np.nanmax(m[np.isfinite(m)]) for m in maps]
    elif isinstance(vmax, (int, float)):
        vmax_value = [vmax,]*nmaps
    else:
        raise ValueError()
    
    vmin_value = np.array(vmin_value)
    vmax_value = np.array(vmax_value)
    
    vmin_label = "min" if vmin=='auto' else vmin_value[0]
    vmax_label = "max" if vmax=='auto' else vmax_value[0]     
    
    if emin+emax>=1:
        raise ValueError()
        
    delta = vmax_value - vmin_value
    
    if vmin=='auto':
        vmin_value = vmin_value - (emin*delta/(1-emin-emax))
    if vmax=='auto':
        vmax_value = vmax_value + (emax*delta/(1-emin-emax))
    
    if not roundto is None:
        vmin_value = np.floor(vmin_value/roundto)*roundto
        vmax_value = np.ceil(vmax_value/roundto)*roundto
    
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=figsize)
    
    ax = np.atleast_2d(ax)

    for k, a in enumerate(ax.ravel()):
        if k>=nmaps:
            a.axis('off')
            continue
        
        if fill:
            a.fill_between(x, maps[k], color=color, alpha=fill_alpha, lw=0)
            
        a.plot(x, maps[k], color=color)
        
        a.set_ylim((vmin_value[k], vmax_value[k]))
        a.set_xlim((x[0], x[-1]))
        
        if vmin=='auto':
            a.text(x[0], vmax_value[k], "{}".format(vmin_value[k]), va='bottom', ha='left')
        if vmax=='auto':
            a.text(x[-1], vmax_value[k], "{}".format(vmax_value[k]), va='bottom', ha='right')
        
        if not a is ax[-1,0]:
            a.set(yticks=[], xticks=[])
        else:
            a.set(ylabel=ylabel, xlabel=xlabel)
            a.set(yticks=np.linspace(vmin_value[k], vmax_value[k], 6),
                  yticklabels=['min' if vmin=='auto' else vmin_value[k]]+['',]*4+['max' if vmax=='auto' else vmax_value[k]],
                  xticks=[x[0], x[-1]] )
    
    return fig, ax

def plot_2d_maps(maps, coordinates=None, labels=None,
                 grid=None, figsize=None,
                 cmap='inferno', cbar=True, cbar_divisions=5, 
                 cmin='shared', cmax='shared', roundto=None):
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
    labels : {'x':'', 'y':'', 'color':''}
        Dictionary with labels for x and y axes and for the color bar
    cbar : bool
        Show color bar.
    cbar_divisions: int
        Number of divisions to show in color bar.
    grid : (rows,cols)
        Number of row and columns for the image grid. Entries can be None,
        which means that the value will be automatically computed.
    figsize : (width, height)
    cmin, cmax : 'shared', 'auto', scalar
        The minimum and maximum color value. If 'shared', the value will be
        computed as the min/max across all *maps*. If 'auto', the value will
        be computed separately for each map and will be displayed above each
        plot (minimum on the top left, maximum on the top right). If a scalar,
        the value will be fixed for all plots.
    roundto : scalar
        Round down/up the minimum and maximum color values to the nearest
        multiple of *roundto*.
    
    Returns
    -------
    fig : Figure
    ax : (nrows, ncols) array of Axes
    
    """
        
    nmaps = len(maps)
    
    if nmaps<1:
        raise ValueError("Expecting at least one 2d map.")
    
    maps = [np.asarray(m) for m in maps]
    
    shape = maps[0].shape
    if len(shape)!=2 or any([m.shape!=shape for m in maps]):
        raise ValueError("All maps should have the same 2d shape.")
    
    if grid is None:
        grid = (None, None)
    elif isinstance(grid, int):
        grid = (None, grid)
    
    nrows, ncols = grid
    
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(nmaps)))
        nrows = (nmaps + (ncols-1))//ncols
    elif nrows is None:
        nrows = (nmaps + (ncols-1))//ncols
    else:
        ncols = (nmaps + (nrows-1))//nrows
    
    if coordinates is None:
        extent = None
    else:
        extent = np.zeros(4)
        for c, slc, nn in zip(
            coordinates, [slice(0,2), slice(2,None)], shape):
            if len(c)==2:
                extent[slc] = c
            elif len(c)==nn:
                cdiff = np.diff(c)
                extent[slc] = [c[0]-cdiff[0]/2, c[-1]+cdiff[-1]/2]
            elif len(c)==nn+1: # bins
                extent[slc] = [c[0], c[-1]]
            else:
                raise ValueError("Incorrect size of coordinates arrays.")
    
    if cmin=='shared':
        cmin_value = [np.nanmin([np.nanmin(m[np.isfinite(m)]) for m in maps]),]*nmaps
    elif cmin=='auto':
        cmin_value = [np.nanmin(m[np.isfinite(m)]) for m in maps]
    elif isinstance(cmin, (int, float)):
        cmin_value = [cmin,]*nmaps
    else:
        raise ValueError()
        
    if cmax=='shared':
        cmax_value = [np.nanmax([np.nanmax(m[np.isfinite(m)]) for m in maps]),]*nmaps
    elif cmax=='auto':
        cmax_value = [np.nanmax(m[np.isfinite(m)]) for m in maps]
    elif isinstance(cmax, (int, float)):
        cmax_value = [cmax,]*nmaps
    else:
        raise ValueError()
    
    if not roundto is None:
        cmin_value = np.floor(np.array(cmin_value)/roundto)*roundto
        cmax_value = np.ceil(np.array(cmax_value)/roundto)*roundto
    
    cmin_label = "min" if cmin=='auto' else cmin_value[0]
    cmax_label = "max" if cmax=='auto' else cmax_value[0]
    
    label_dict = {'x':'', 'y':'', 'color':''}
    
    if not labels is None:
        label_dict.update(labels)
        
    
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=figsize)
    
    ax = np.atleast_2d(ax)

    for k, a in enumerate(ax.ravel()):
        if k>=nmaps:
            a.axis('off')
            continue
        
        img = a.imshow(maps[k].T, interpolation='none', aspect='auto',
                       origin='lower', extent=extent, cmap=cmap);
        
        img.set_clim( (cmin_value[k], cmax_value[k]) )
        
        if cmin=='auto':
            a.text(extent[0], extent[3], "{}".format(cmin_value[k]), va='bottom', ha='left')
        if cmax=='auto':
            a.text(extent[1], extent[3], "{}".format(cmax_value[k]), va='bottom', ha='right')
            
        if not a is ax[-1,0]:
            a.set(yticks=[], xticks=[])
        else:
            a.set(ylabel=label_dict['y'], xlabel=label_dict['x'])
    

        if cbar and k==0:
            
            cax = plt.colorbar(img, ax=ax)
            cax.set_label(label_dict['color'])
            
            if cbar_divisions is None or cbar_divisions<1:
                cax.set_ticks([])
            else:
                cax.set_ticks(cmin_value[k] + cmax_value[k]*np.arange(cbar_divisions+1)/cbar_divisions)
            
            cax.set_ticklabels([])
            cax.ax.grid('y', color='w')
            
            cax.ax.tick_params(length=0)
            cax.ax.text(np.mean(cax.ax.get_xlim()),cmin_value[k], cmin_label,va='top', ha='center')
            cax.ax.text(np.mean(cax.ax.get_xlim()),cmax_value[k], cmax_label,va='bottom', ha='center')
    
    return fig, ax


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
    kwargs : EventCollection or FastRaster options

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
                FastRaster(
                    events[k], foreground_color=color, foreground_alpha=alpha, **kwargs
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
    kwargs : FastRaster options

    Returns
    -------
    collection of rasters

    """

    kwargs["kind"] = "raster"
    return plot_events(events, **kwargs)


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
    """Add static scalebar to axes

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
    """Add message artist to axes

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
    """Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    Parameters
    ----------
    ax : the axis to attach ticks to
    matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    hidex,hidey : if True, hide x-axis and y-axis of parent
    **kwargs : additional arguments passed to AnchoredScaleBars

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
