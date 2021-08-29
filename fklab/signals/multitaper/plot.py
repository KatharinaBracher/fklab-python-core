"""
Plots
=====

"""

__all__ = ["plot_spectrum", "plot_spectrogram", "plot_coherence", "plot_coherogram"]

import matplotlib.pyplot as plt
import numpy as np
from .multitaper import mtspectrum, mtcoherence, mtcoherogram, mtspectrogram


def plot_spectrum(
    data, t=None, axes=None, units=None, db=True, color="black", **kwargs
):
    """Plot power spectral density of data vector.

    Parameters
    ----------
    data : 1d array
    t : 1d array, optional
        Time vector. Start time and sampling frequency are automatically
        calculated from the time vector.
    axes : Axes, optional
        Destination axes.
    units : str, optional
        Units of the data (e.g. mV)
    db : bool, optional
        Plot density in dB.
    color : any matplotlib color, optional
        Color of plot.
    kwargs : mtspectrum parameters

    Returns
    -------
    axes : matplotlib Axes object
    artists : list of plot elements
    (s,f,err,options) : output of mtspectrum

    """
    if not t is None:
        kwargs["start_time"] = kwargs.get("start_time", t[0])
        kwargs["fs"] = kwargs.get("fs", 1.0 / np.mean(np.diff(t)))

    S, f, err, options = mtspectrum(data, **kwargs)

    if db:
        S = 10.0 * np.log10(S)
        if not err is None:
            err = 10.0 * np.log10(err)

    if axes is None:
        axes = plt.gca()

    artists = []

    if not err is None:
        if err.ndim == 2:  # average case
            artists.append(axes.fill_between(f, err[0, :], err[1, :], facecolor=color, alpha=0.2))
        else:
            artists.append(axes.fill_between(f, err[0, :, 0], err[1, :, 0], facecolor=color, alpha=0.2))

    artists.extend(axes.plot(f, S, color=color))

    if units is None or units == "":
        units = "1"
    else:
        units = str(units)
        units = units + "*" + units

    axes.set(xlabel="frequency [Hz]")
    axes.set(ylabel="power spectral density [{units}/Hz] {db}".format(
            units=units, db="in db" if db else ""
        )
    )

    return axes, artists, (S, f, err, options)


def plot_spectrogram(
    data, t=None, axes=None, units=None, db=True, colorbar=True, **kwargs
):
    """Plot spectrogram of data vector.

    Parameters
    ----------
    data : 1d array
    t : 1d array, optional
        Time vector. Start time and sampling frequency are automatically
        calculated from the time vector.
    axes : Axes, optional
        Destination axes.
    units : str, optional
        Units of the data (e.g. mV)
    db : bool, optional
        Plot density in dB.
    kwargs : mtspectrogram parameters

    Returns
    -------
    axes : matplotlib Axes object
    artists : plot elements
    (s,t,f,err,options) : output of mtspectrogram

    """
    if not t is None:
        kwargs["start_time"] = kwargs.get("start_time", t[0])
        kwargs["fs"] = kwargs.get("fs", 1.0 / np.mean(np.diff(t)))

    winsize = kwargs.get("window_size")

    S, t, f, err, options = mtspectrogram(data, **kwargs)

    if db:
        S = 10.0 * np.log10(S)

    if axes is None:
        axes = plt.gca()

    artists = []

    artists.append(
        axes.imshow(
            S.T,
            cmap="YlOrRd",
            aspect="auto",
            origin="lower",
            extent=[t[0, 0], t[-1, 1], f[0], f[-1]],
            interpolation="nearest",
        )
    )

    axes.set_ylabel("frequency [Hz]")
    axes.set_xlabel(
        "{label} [s]".format(
            label="time" if kwargs.get("triggers", None) is None else "latency"
        )
    )

    if colorbar:
        cbar = plt.colorbar(artists[0], ax=axes)
        artists.append(cbar)

        if units is None or units == "":
            units = "1"
        else:
            units = str(units)
            units = units + "*" + units

        cbar.set_label(
            "power spectral density [{units}/Hz] {db}".format(
                units=units, db="in db" if db else ""
            )
        )

    plt.draw()

    return axes, artists, (S, t, f, err, options)


def plot_coherence(signal1, signal2, t=None, axes=None, color="black", **kwargs):
    """Plot coherence between two data vectors.

    Parameters
    ----------
    signal1 : 1d array
    signal2 : 1d array
    t : 1d array, optional
        Time vector. Start time and sampling frequency are automatically
        calculated from the time vector.
    axes : Axes, optional
        Destination axes
    color : any matplotlib color, optional
        Color of plot.
    kwargs : mtcoherence parameters

    Returns
    -------
    axes : matplotlib Axes object
    artists : plot elements
    (coh,phi,f,err,options) : output of mtcoherence

    """
    if not t is None:
        kwargs["start_time"] = kwargs.get("start_time", t[0])
        kwargs["fs"] = kwargs.get("fs", 1.0 / np.mean(np.diff(t)))

    coh, phi, f, err, options = mtcoherence(signal1, signal2, **kwargs)

    if axes is None:
        axes = plt.gca()

    artists = []

    if not err[2] is None:
        artists.append(
            axes.fill_between(
                f, err[2][0, :, 0], err[2][1, :, 0], facecolor=color, alpha=0.2
            )
        )

    artists.extend(axes.plot(f, coh, color=color))

    if not err[0] is None:
        artists.append(axes.axhline(y=err[0], color="red", linestyle=":"))

    axes.set(xlabel="frequency [Hz]", ylabel="coherence")

    axes.set(ylim=(0, 1))

    return axes, artists, (coh, phi, f, err, options)


def plot_coherogram(signal1, signal2, t=None, axes=None, **kwargs):
    """Plot coherogram of two data vectors.

    Parameters
    ----------
    signal1 : 1d array
    signal2 : 1d array
    t : 1d array, optional
        Time vector. Start time and sampling frequency are automatically
        calculated from the time vector.
    axes : Axes, optional
        Destination axes
    kwargs : mtcoherogram parameters

    Returns
    -------
    axes : matplotlib Axes object
    artists : plot elements
    (coh,phi,t,f,err,options) : output of mtcoherence

    """
    if not t is None:
        kwargs["start_time"] = kwargs.get("start_time", t[0])
        kwargs["fs"] = kwargs.get("fs", 1.0 / np.mean(np.diff(t)))

    winsize = kwargs.get("window_size")

    coh, phi, t, f, err, options = mtcoherogram(signal1, signal2, **kwargs)

    if axes is None:
        axes = plt.gca()

    artists = []

    artists.append(
        axes.imshow(
            coh.T,
            cmap="YlOrRd",
            aspect="auto",
            origin="lower",
            extent=[t[0, 0], t[-1, 1], f[0], f[-1]],
            interpolation="nearest",
            vmin=0, vmax=1.0
        )
    )
    
    axes.set(ylabel="frequency [Hz]")
    axes.set(xlabel="{label} [s]".format(
            label="time" if kwargs.get("triggers", None) is None else "latency"
        )
    )

    cbar = plt.colorbar(artists[0], ax=axes)
    cbar.set_label("coherence")

    artists.append(cbar)

    plt.draw()

    return axes, artists, (coh, phi, t, f, err, options)
