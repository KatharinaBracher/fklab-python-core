"""
====================================================
Polar plot utilities (:mod:`fklab.plot.plots.polar`)
====================================================

.. currentmodule:: fklab.plot.plots.polar

Function for polar plots.

"""
import fractions

import matplotlib.projections.polar
import matplotlib.pyplot as plt
import matplotlib.ticker
import mpl_toolkits.axes_grid1.inset_locator
import numpy as np

import fklab.statistics.circular
from .utilities import setup_axes_grid

__all__ = [
    "polar_colorbar",
    "plot_polar_map",
    "set_polar_axes",
    "setup_polar_axes_grid",
    "float_radial_axis",
    "style_polar_axes",
    "annotate_angles",
]


def polar_colorbar(cax=None, cmap="hsv", rotation=0, **kwargs):
    """Plot circular colorbar.

    Parameters
    ----------
    cax : Axes
        Colorbar axes.
    cmap : colormap
    rotation : float
    **kwargs :
        Extra keyword arguments that are passed to *setup_polar_axes*.

    Returns
    -------
    Axes

    """
    theta = np.linspace(0 + rotation, 2 * np.pi + rotation, 1001)
    data = theta[None, :]

    cax, _ = plot_polar_map(
        data, theta=theta, rho=[0, 1], cmap=cmap, ax=cax, vmin=theta[0], vmax=theta[-1]
    )

    if not "show_radial_axis" in kwargs:
        kwargs["show_radial_axis"] = False

    set_polar_axes(cax, **kwargs)


def plot_polar_map(
    data, theta=None, rho=None, cmap="inferno", ax=None, vmin=None, vmax=None
):
    """Plot a polar map.

    Parameters
    ----------
    data : (n,m) array
        Data for map. The radial axis runs along the first dimension and the
        angular axis runs along the second dimensions of the array.
    theta : (m,) array
        Array with angular coordinates.
    rho : (n,) array
        Array with radial coordinates.
    cmap : colormap
    ax : None or Axes
        Destination polar axes. If None, a new axes wil be created.
    vmin, vmax : None or float
        Color limits of the map. If None, the limit will be determined from
        the data.

    Returns
    -------
    ax : Axes
    mesh : Mesh

    """
    if ax is None:
        ax = plt.subplot(projection="polar")

    data = np.asarray(data)

    if not data.ndim == 2:
        raise ValueError("Expecting 2D data array.")

    if vmin is None or vmin == "auto":
        vmin = np.nanmin(data)

    if vmax is None or vmax == "auto":
        vmax = np.nanmax(data)

    nrho, ntheta = data.shape

    if theta is None:
        theta = 2 * np.pi * (np.arange(ntheta + 1) - 0.5) / ntheta
    else:
        theta = np.atleast_1d(theta)

        if not theta.ndim == 1:
            raise ValueError("Expecting 1D theta array.")

        if len(theta) == ntheta:
            # theta defines centers, build edge vector instead
            # theta is required to be sorted
            delta = fklab.statistics.circular.diff(theta) / 2.0
            theta = np.concatenate(
                ([theta[0] - delta[0]], theta[:-1] + delta, [theta[-1] + delta[-1]])
            )

    if rho is None:
        rho = np.arange(nrho + 1) - 0.5
    else:
        rho = np.atleast_1d(rho)

        if not rho.ndim == 1:
            raise ValueError("Expecting 1d rho array.")

        if len(rho) == nrho:
            # rho defines centers, build edge vector instead
            # rho is required to be sorted
            delta = np.diff(rho) / 2.0
            rho = np.concatenate(
                ([rho[0] - delta[0]], rho[:-1] + delta, [rho[-1] + delta[-1]])
            )

    distance, angle = np.meshgrid(rho, theta)

    mesh = ax.pcolormesh(angle, distance, data.T, cmap=cmap)
    mesh.set_clim((vmin, vmax))

    return ax, mesh


def set_polar_axes(axes, **kwargs):
    """Set polar axes properties.

    Parameters
    ----------
    axes : PolarAxes or iterable of PolarAxes
    style : str
    style_kw : dict
    show_radial_axis : bool
    show_angle_axis : bool
    float_radial_axis : bool
    float_radial_kw : dict
    radial_min, radial_max : float
    annulus : float
    radial_grid : bool
    angle_grid : bool
    radial_ticks : iterable
    angle_ticks : iterable
    radial_ticklabels: iterable
    angle_ticklabels : iterable
    radial_tick_params : dict
    angle_tick_params : dict

    """
    if not isinstance(axes, (tuple, list, np.ndarray)):
        axes = [axes]

    if isinstance(axes, np.ndarray):
        axes = axes.ravel()

    for ax in axes:

        if not hasattr(ax, "extra_properties"):
            setattr(ax, "extra_properties", {})

        props = ax.extra_properties

        if "style" in kwargs:
            style_polar_axes(ax, style=kwargs["style"], **kwargs.get("style_kw", {}))

        if "show_radial_axis" in kwargs:
            ax.yaxis.set_visible(kwargs["show_radial_axis"])

        if "show_angle_axis" in kwargs:
            ax.xaxis.set_visible(kwargs["show_angle_axis"])

        flax = props.get("float_radial_axis", None)

        if "float_radial_axis" in kwargs:
            if kwargs["float_radial_axis"]:
                if not flax is None:
                    pass
                else:
                    props["float_radial_axis"] = flax = float_radial_axis(
                        ax, **kwargs.get("float_radial_kw", {})
                    )
            else:
                if not flax is None:
                    flax.remove()
                    props["float_radial_axis"] = flax = None
                else:
                    pass

        radial_min = kwargs.get("radial_min", None)
        radial_max = kwargs.get("radial_max", None)

        ax.set_ylim(radial_min, radial_max)

        if not flax is None:
            flax.set(xlim=(radial_min, radial_max))

        annulus = None
        if "annulus" in kwargs:
            annulus = float(kwargs["annulus"])
        elif "annulus" in props:
            annulus = props["annulus"]

        if not annulus is None:
            radial_min, radial_max = ax.get_ylim()
            origin = radial_min - 2 * annulus * (radial_max - radial_min)
            ax.set_rorigin(origin)
            props["annulus"] = annulus

        # update location of floating radial axis
        if not flax is None:
            origin = ax.get_rorigin()
            rlim = ax.get_ylim()

            x = 0.5 + 0.5 * (rlim[0] - origin) / (rlim[1] - origin)

            ip = mpl_toolkits.axes_grid1.inset_locator.InsetPosition(
                ax, [x, 0, 1 - x, 1]
            )
            flax.set_axes_locator(ip)

        if "radial_label" in kwargs:
            if not flax is None:
                flax.set_xlabel(kwargs["radial_label"])

        if "radial_grid" in kwargs:
            ax.yaxis.grid(kwargs["radial_grid"])

        if "angle_grid" in kwargs:
            ax.xaxis.grid(kwargs["angle_grid"])

        if "radial_ticks" in kwargs:
            ax.yaxis.set_ticks(kwargs["radial_ticks"])
            if not flax is None:
                flax.xaxis.set_ticks(kwargs["radial_ticks"])

        if "angle_ticks" in kwargs:
            ax.xaxis.set_ticks(kwargs["angle_ticks"])

        if "radial_ticklabels" in kwargs:
            ax.yaxis.set_ticklabels(kwargs["radial_ticklabels"])
            if not flax is None:
                flax.xaxis.set_ticklabels(kwargs["radial_ticklabels"])

        if "angle_ticklabels" in kwargs:
            ax.xaxis.set_ticklabels(kwargs["angle_labels"])

        if "radial_tick_params" in kwargs:
            ax.tick_params(axis="y", **kwargs["radial_tick_params"])
            if not flax is None:
                flax.tick_params(axis="x", **kwargs["radial_tick_params"])

        if "angle_tick_params" in kwargs:
            ax.tick_params(axis="x", **kwargs["angle_tick_params"])


def setup_polar_axes_grid(
    n,
    show_reference=False,
    reference_kw={},
    float_radial_kw={},
    axes_props={},
    **kwargs,
):
    """Construct figure with grid of polar axes.

    Parameters
    ----------
    n : int
        The number of polar axes.
    show_reference : bool
        Create extra polar axes at the end with an angle reference and
        floating radial axis.
    reference_kw : dict
        Extra keyword arguments for *plot_angle_reference*.
    float_radial_kw : dict
        Extra keyword arguments for *float_radial_axis*.
    grid_kw : dict
    **kwargs

    Returns
    -------
    fig : Figure
    ax : array of Axes

    """
    ntotal = int(n) + int(show_reference)

    fig, axes = setup_axes_grid(ntotal, subplot_kw={"projection": "polar"}, **kwargs)

    if not "show_angle_axis" in axes_props:
        axes_props["show_angle_axis"] = not show_reference

    if not "show_radial_axis" in axes_props:
        axes_props["show_radial_axis"] = not show_reference

    set_polar_axes(axes, **axes_props)

    for k, ax in enumerate(axes.ravel()):

        if k == axes.size - 1 and show_reference:
            ax.axis("on")
            annotate_angles(ax, **reference_kw)
            float_radial_axis(ax, axes_props.get("radial_label", ""), **float_radial_kw)
        elif k >= n:
            ax.axis("off")

    return fig, axes


def float_radial_axis(host, label="", offset=5, y=0, tickprops={}, labelprops={}):
    """Create floating radial axis for polar plot.

    Parameters
    ----------
    host : Axes
        Polar axes to which a floating radial axis will be attached.
    label : str
        Radial axis label.
    offset : scalar
        Vertical offset of axis.
    y : float
        Vertical position of radial axis (in normalized axes coordinates)
    tickprops : dict
        Properties of ticks and tick labels.
    labelprops : dict
        Properties of axis label.

    Returns
    -------
    Axes

    """
    if not hasattr(host, "extra_properties"):
        setattr(host, "extra_properties", {})

    props = host.extra_properties
    if "float_radial_axis" in props:
        props["float_radial_axis"].pop().remove()

    origin = host.get_rorigin()
    rlim = host.get_ylim()

    x = 0.5 + 0.5 * (rlim[0] - origin) / (rlim[1] - origin)

    ax = host.inset_axes([x, y, 1 - x, 1 - y])

    ax.set(xlim=rlim)
    ax.xaxis.set_major_locator(host.yaxis.get_major_locator())

    for side in ["left", "right", "top"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_position(("outward", offset))

    ax.yaxis.set_visible(False)
    ax.set_xlabel(label, **labelprops)
    ax.set_facecolor("none")

    ax.tick_params(axis="x", **tickprops)

    host.tick_params(axis="y", labelcolor="none")

    props["float_radial_axis"] = ax

    return ax


def radian_pi_formatter(limit=10):
    """Angle label formatter.

    Formats angle labels as fractions of pi.

    Parameters
    ----------
    limit : int
        The largest value of the fraction denominator.

    Returns
    -------
    Formatter

    """

    def _format(x, pos=None):
        rad_x = x / np.pi
        if rad_x == 0:
            return "0"
        elif rad_x == 1:
            return "π"
        frac = fractions.Fraction(rad_x).limit_denominator(limit)

        return fr"$\frac{{{frac.numerator}}}{{{frac.denominator}}}$π"

    return matplotlib.ticker.FuncFormatter(_format)


def compass_formatter(upper=True):
    """Angle label formatter.

    Formats angle labels as compass directions.

    Parameters
    ----------
    upper : bool
        Labels are upper case.

    Returns
    -------
    Formatter

    """
    _map = {
        0: "N",
        45: "NE",
        90: "E",
        135: "SE",
        180: "S",
        225: "SW",
        270: "W",
        315: "NW",
        360: "N",
    }

    def _format(x, pos=None):
        x = int(180 * x / np.pi)

        if x < 0:
            x = x + 360

        s = _map.get(x, "")

        if not upper:
            s = s.tolower()

        return s

    return matplotlib.ticker.FuncFormatter(_format)


def egocentric_formatter(labels=None):
    """Angle label formatter.

    Formats angle labels as egocentric coordinates.

    Parameters
    ----------
    labels : dict
        Dictionary that maps (integer) degrees to label. By default, 0 degrees
        is "front", 90 degrees is "left", 180 degrees is "back" and 270 degrees
        is "right".

    Returns
    -------
    Formatter

    """
    _map = {0: "front", 90: "left", 180: "back", 270: "right"}

    if not labels is None:
        _map.update(labels)

    def _format(x, pos=None):

        x = int(180 * x / np.pi)

        if x < 0:
            x = x + 360

        if x == 360:
            x = 0

        return _map.get(x, "")

    return matplotlib.ticker.FuncFormatter(_format)


def default_formatter():
    """Angle label formatter.

    Formats angle labels as degrees.

    Returns
    -------
    Formatter

    """
    return matplotlib.projections.polar.ThetaFormatter()


# Dictionary of polar axes styles
POLAR_AXES_STYLES = {
    "math": dict(theta_offset=0, theta_direction=1, formatter=radian_pi_formatter),
    "math-degrees": dict(
        theta_offset=0, theta_direction=1, formatter=default_formatter
    ),
    "navigation": dict(
        theta_offset=0.5 * np.pi, theta_direction=-1, formatter=radian_pi_formatter
    ),
    "navigation-degrees": dict(
        theta_offset=0.5 * np.pi, theta_direction=-1, formatter=default_formatter
    ),
    "compass": dict(
        theta_offset=0.5 * np.pi, theta_direction=-1, formatter=compass_formatter
    ),
    "egocentric": dict(
        theta_offset=0.5 * np.pi, theta_direction=1, formatter=egocentric_formatter
    ),
}


def style_polar_axes(ax, style="math", **kwargs):
    """Set polar axes style.

    This function applies a predefined style to a polar axes. The properties
    that are set are: the zero-location and direction (clock-wise or
    counter-clock-wise) of the angular axis, the formatting of the angle
    labels.

    Parameters
    ----------
    ax : Axes
        Polar axes instance.
    style : str
        The style to apply. One of: "math", "math-degrees", "navigation",
        "navigation-degrees", "compass", "egocentric".
    **kwargs
        Extra keyword arguments that are passed to the formatter.
    """
    if style is None or style == "default":
        style = "math"

    style = POLAR_AXES_STYLES[style]

    if "formatter" in style:
        ax.xaxis.set_major_formatter(style["formatter"](**kwargs))

    if "theta_offset" in style:
        ax.set_theta_offset(style["theta_offset"])

    if "theta_direction" in style:
        ax.set_theta_direction(style["theta_direction"])


def annotate_angles(
    ax,
    angles=None,
    facecolor="white",
    radial_fraction=0.6,
    arrowprops={},
    annotation_kw={},
    **kwargs,
):
    """Plot angular reference.

    Parameters
    ----------
    ax : Axes
        Polar axes to draw windrose in.
    angles : 1d array-like
        Angles (in radians) at which labels are drawn.
    facecolor : color
        Background color of the axes.
    radial_fraction : float
        Radial location of the labels, as fraction of the radial limits.
    arrowprops: dict
        Keyword arguments to set additional arrow properties in *annotation*
        call.
    annotation_kw : dict
        Keyword arguments to set additional annotation properties in *annotation*
        call.
    **kwargs
        Extra keyword arguments for *set_polar_axes* (e.g. to polar plot style)

    """
    set_polar_axes(ax, **kwargs)

    ax.grid(False)
    ax.set_facecolor(facecolor)
    ax.yaxis.set(visible=False)
    ax.xaxis.set(visible=False)

    if angles is None:
        angles = np.arange(4) * np.pi / 2

    fmt = ax.xaxis.get_major_formatter()
    labels = [fmt(x) for x in angles]

    rmin, rmax = ax.get_ylim()

    arrow = dict(color="black", arrowstyle="<-")
    arrow.update(arrowprops)

    for a, s in zip(angles, labels):
        ax.annotate(
            s,
            [a, rmin],
            [a, rmin + (rmax - rmin) * radial_fraction],
            ha="center",
            va="center",
            arrowprops=arrow,
            annotation_clip=False,
            **annotation_kw,
        )
