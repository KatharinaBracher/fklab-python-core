"""
=============================================
Segment plotting (:mod:`fklab.segments.plot`)
=============================================

.. currentmodule:: fklab.segments.plot

Provides functions for plotting segments.

"""
import matplotlib.collections
import matplotlib.container
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np

from .basic_algorithms import check_segments

__all__ = [
    "plot_segments_as_lines",
    "plot_segments_as_patches",
    "SegmentLinesContainer",
]


class SegmentLinesContainer(matplotlib.container.Container):
    def __init__(self, artists, **kwargs):
        lines, startmarkers, endmarkers = artists
        self.lines = lines
        self.startmarkers = startmarkers
        self.endmarkers = endmarkers
        super().__init__(artists, **kwargs)


def plot_segments_as_lines(
    x,
    y=0,
    ycoords="data",
    orientation="horizontal",
    marker=None,
    startmarker=None,
    endmarker=None,
    markersize=None,
    color="black",
    alpha=1.0,
    axes=None,
    autoscale=True,
    lineprops=None,
):
    """Plot segments as lines with optional start and end markers.

    Parameters
    ----------
    x : (n,2) array-like
        Start and end points of the segments.
    y : float
        Coordinate in the y-axis of the plotted lines.
    ycoords : str
        Coordinate system of the y-coordinate. Either 'data' coordinates
        or 'axes' coordinates.
    orientation : str
        Either 'horizontal' or 'vertical'. If 'vertical', then the segment
        start and end points in `x` are oriented along the y-axis and
        `y` is interpreted as an offset along the x-axis.
    marker, startmarker, endmarker : marker specification
        Valid matplotlib marker specification. If `startmarker` or
        `endmarker` is None, then the value from `marker` is taken.
    markersize : float
        Size of the start and end markers.
    color : color specification
        Valid matplotlib color specification for lines and markers.
    alpha : float
        Transparency for lines and markers.
    axes : None or matplotlib Axes
    lineprops : None or dict
        Dictionay with additional line properties that is passed to
        `matplotlib.collections.LineCollection` constructor.

    Returns
    -------
    SegmentLinesContainer

    """

    x = check_segments(x)
    y = float(y)

    if not ycoords in ("data", "axes"):
        raise ValueError(
            "Unrecognized value for ycoords parameter. Valid values are 'data' and 'axes'."
        )

    if not orientation in ("horizontal", "vertical"):
        raise ValueError(
            "Unrecognized value for orientation parameter. Valid values are 'horizontal' and 'vertical'."
        )

    startmarker = marker if startmarker is None else startmarker
    endmarker = marker if endmarker is None else endmarker

    _lineprops = dict()
    if not lineprops is None:
        _lineprops.update(lineprops)

    if axes is None:
        axes = plt.gca()

    tform = axes.transData

    if ycoords == "axes":
        if orientation == "vertical":
            tform = axes.get_yaxis_transform()
        else:
            tform = axes.get_xaxis_transform()

    # prepare line data
    if orientation == "horizontal":
        lines = [((start, y), (end, y)) for start, end in x]
    else:
        lines = [((y, start), (y, end)) for start, end in x]

    linecollection = matplotlib.collections.LineCollection(
        lines, color=color, alpha=alpha, transform=tform, **_lineprops
    )

    axes.add_collection(linecollection, autolim=autoscale)

    y = np.full(len(x), y)

    if not startmarker is None:
        startmarkers, = axes.plot(
            x[:, 0] if orientation == "horizontal" else y,
            y if orientation == "horizontal" else x[:, 0],
            color=color,
            alpha=alpha,
            linestyle="none",
            markersize=markersize,
            marker=startmarker,
            transform=tform,
        )

    if not endmarker is None:
        endmarkers, = axes.plot(
            x[:, 1] if orientation == "horizontal" else y,
            y if orientation == "horizontal" else x[:, 1],
            color=color,
            alpha=alpha,
            linestyle="none",
            markersize=markersize,
            marker=endmarker,
            transform=tform,
        )

    if autoscale:
        axes.autoscale_view()

    return SegmentLinesContainer((linecollection, startmarkers, endmarkers))


def plot_segments_as_patches(
    x,
    y=0,
    height=1,
    anchor=None,
    ycoords="data",
    orientation="horizontal",
    axes=None,
    **kwargs
):
    """Plot segments as filled rectangular patches.

    Parameters
    ----------
    x : (n,2) array-like
        Start and end points of the segments.
    y : float
        Anchor point offset along the y-axis.
    height : float
        Height of the rectangles.
    anchor : float
        Anchoring point of the rectangle along the y-axis. This value ranges
        from 0 (bottom) to 1 (top). The final y-offset is computed as
        `y - anchor * height`.
    ycoords : str
        Coordinate system of the y-coordinate. Either 'data' coordinates
        or 'axes' coordinates. If 'data', then the default anchoring point
        is 0.5, such that `y` represents the center of the drawn rectangles.
        If `axes`, then default anchoring point is 0, such that `y`
        represents the offset of the bottom of the drawn rectangles.
    orientation : str
        Either 'horizontal' or 'vertical'. If 'vertical', then the segment
        start and end points in `x` are oriented along the y-axis and
        `y` is interpreted as an offset along the x-axis.
    axes : None or matplotlib Axes
    **kwargs :
        Extra keyword arguments that are passed to matplotlib's
        `broken_barh` function.

    Returns
    -------
    BrokenBarHCollection

    """

    x = check_segments(x)
    y = float(y)
    height = float(height)

    if not ycoords in ("data", "axes"):
        raise ValueError(
            "Unrecognized value for ycoords parameter. Valid values are 'data' and 'axes'."
        )

    if not orientation in ("horizontal", "vertical"):
        raise ValueError(
            "Unrecognized value for orientation parameter. Valid values are 'horizontal' and 'vertical'."
        )

    if axes is None:
        axes = plt.gca()

    tform = matplotlib.transforms.Affine2D()

    if orientation == "vertical":
        tform = tform.rotate_deg(90).scale(-1, 1)

    if ycoords == "axes":
        if orientation == "vertical":
            tform = tform + axes.get_yaxis_transform()
        else:
            tform = tform + axes.get_xaxis_transform()
    else:
        tform = tform + axes.transData

    if anchor is None:
        anchor = 0 if ycoords == "axes" else 0.5

    anchor = float(anchor)

    return axes.broken_barh(
        np.column_stack([x[:, 0], x[:, 1] - x[:, 0]]),
        (y - height * anchor, height),
        transform=tform,
        **kwargs
    )
