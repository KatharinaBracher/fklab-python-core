"""
===========================================
Filter (:mod:`fklab.signals.filter.filter`)
===========================================

.. currentmodule:: fklab.signals.filter.filter

Utilities for the design, application and inspection of digital filters.

Filter utilities
================
"""
import warnings
from functools import reduce

import matplotlib.pyplot as plt
import matplotlib.transforms
import mpl_toolkits.axes_grid1
import numpy as np
import pandas
import scipy.signal
from matplotlib import gridspec

import fklab.signals.smooth
from fklab.version._core_version._version import __version__

__all__ = [
    "standard_frequency_bands",
    "construct_filter",
    "construct_low_pass_filter",
    "construct_high_pass_filter",
    "apply_filter",
    "apply_median_filter",
    "apply_low_pass_filter",
    "apply_high_pass_filter",
    "inspect_filter",
    "plot_filter_amplitude",
    "plot_filter_phase",
    "plot_filter_group_delay",
    "compute_envelope",
    "compute_sliding_rms",
    "contrast_frequency_bands",
]

standard_frequency_bands = {
    "slow": [0.1, 1.0],
    "delta": [1.0, 4.0],
    "theta": [6.0, 12.0],
    "spindle": [7.0, 14.0],
    "beta": [15.0, 30.0],
    "gamma": [30.0, 140.0],
    "gamma_low": [30.0, 50.0],
    "gamma_high": [60.0, 140.0],
    "ripple": [140.0, 225.0],
    "mua": [300.0, 2000.0],
    "spikes": [500.0, 5000.0],
}


def construct_filter(band, fs=1.0, transition_width="25%", attenuation=60):
    """Construct FIR high/low/band-pass filter.

    Parameters
    ----------
    band : str, scalar or 2-element sequence
        either a valid key into the default_frequency_bands dictionary,
        a scalar for a low-pass filter, or a 2-element sequence with lower
        and upper pass-band frequencies. Use 0., None, Inf or NaN for the
        lower/upper cut-offs in the sequence to define a low/high-pass filter.
        If band[1]<band[0], then a stop-band filter is constructed.
    fs : scalar, optional
        sampling frequency of the signal to be filtered
    transition_width : str or scalar, optional
        size of the transition between pass and stop bands. Can be either
        a scalar frequency or a string that represents a transition width
        relative to the size of the pass band (e.g. "25%", the percentage
        sign is required).
    attenuation : scalar, optional
        stop-band attenuation in dB.

    Returns
    -------
    1D array
        filter coefficients

    """
    # look up pre-defined frequency band
    if isinstance(band, str):
        band = standard_frequency_bands[band]

    band = np.array(band, dtype=np.float64).ravel()

    if len(band) == 1:
        # scalar -> low=pass filter
        band = np.array([0.0, float(band)], dtype=np.float64)
    elif len(band) != 2:
        raise ValueError("Invalid frequency band")

    if np.diff(band) == 0.0:
        raise ValueError("Identical frequencies not allowed.")

    lower, upper = np.logical_or.reduce((np.isnan(band), np.isinf(band), band <= 0.0))

    if not lower and upper:
        # high pass filter
        band = band[0]
        pass_zero = False
        band_width = fs / 2.0 - band
    elif not upper and lower:
        # low pass filter
        band = band[1]
        pass_zero = True
        band_width = band
    elif lower and upper:
        raise ValueError("Invalid frequency band")
    else:
        pass_zero = np.diff(band) < 0
        if len(pass_zero) == 1:
            pass_zero = pass_zero[0]

        if pass_zero:
            band = band[::-1]
        band_width = np.diff(band)

    if fs <= 2 * np.max(band):
        raise ValueError("Frequency band too high for given sampling frequency")

    if isinstance(transition_width, str):
        transition_width = band_width * float(transition_width.rstrip("%")) / 100.0

    N, beta = scipy.signal.kaiserord(attenuation, transition_width * 2.0 / fs)

    # always have odd N
    N = N + (N + 1) % 2

    h = scipy.signal.firwin(
        N, band, window=("kaiser", beta), pass_zero=pass_zero, scale=False, nyq=fs / 2.0
    )

    return h


def construct_low_pass_filter(cutoff, **kwargs):
    """Construct FIR low-pass filter.

    Parameters
    ----------
    cutoff : scalar
        Cut-off frequency for low-pass filter.
    fs : scalar, optional
        sampling frequency of the signal to be filtered
    transition_width : str or scalar, optional
        size of the transition between pass and stop bands. Can be either
        a scalar frequency or a string that represents a transition width
        relative to the size of the pass band (e.g. "25%", the percentage
        sign is required).
    attenuation : scalar, optional
        stop-band attenuation in dB.

    Returns
    -------
    1D array
        filter coefficients

    """
    return construct_filter([None, float(cutoff)], **kwargs)


def construct_high_pass_filter(cutoff, **kwargs):
    """Construct FIR high-pass filter.

    Parameters
    ----------
    cutoff : scalar
        Cut-off frequency for high-pass filter.
    fs : scalar, optional
        sampling frequency of the signal to be filtered
    transition_width : str or scalar, optional
        size of the transition between pass and stop bands. Can be either
        a scalar frequency or a string that represents a transition width
        relative to the size of the pass band (e.g. "25%", the percentage
        sign is required).
    attenuation : scalar, optional
        stop-band attenuation in dB.

    Returns
    -------
    1D array
        filter coefficients

    """
    return construct_filter([float(cutoff), None], **kwargs)


def _filter(b, signal, axis=0, pad=True):
    # b is 1D filter kernel with odd length (e.g. from construct_filter)
    # signal is nD signal, with n>=1
    # filter will be applied along axis
    # odd type reflective padding is used

    # expand dimensionality of b to match signal dimensionality
    if signal.ndim > 1:
        d = np.arange(signal.ndim)
        d = [k for k in d if k != d[axis]]
        b = np.expand_dims(b, axis=d)

    # pad along axis
    if pad:
        n = int((len(b) - 1) / 2)
        pad_width = [(0, 0)] * signal.ndim
        pad_width[axis] = (n, n)
        signal = np.pad(signal, pad_width, mode="reflect", reflect_type="odd")

    # convolve signal with filter kernel
    signal = scipy.signal.convolve(signal, b, mode="valid" if pad else "same")

    return signal


def apply_filter(signal, band, fs=1.0, axis=-1, **kwargs):
    """Apply low/high/band-pass FIR filter to signal.

    Parameters
    ----------
    signal : array
    band : str, scalar or 2-element sequence
        frequency band, either as a string, a scalar or [low,high] sequence.
    fs : scalar, optional
        sampling frequency
    axis : scalar, optional
        axis along which to filter
    kwargs: parameters for construct_filter

    See Also
    --------
    construct_filter

    Returns
    -------
    array
        filtered signal

    Notes
    -----
    The new implementation does not perform a backward filter pass like that of filtfilt, because for (symmetrical) FIR filters this is not necessary. This means that there will be a difference in the output from before version 1.8, as the filter is applied once. One could obtain the same effect as before by calling apply_filter twice, but in practice (with a well designed filter), this should not be necessary.

    """

    b = construct_filter(band, fs=fs, **kwargs)

    if isinstance(signal, (tuple, list)):
        signal = [_filter(b, np.asarray(x), axis=axis) for x in signal]
    else:
        signal = np.asarray(signal)
        signal = _filter(b, signal, axis=axis)

    return signal


def apply_median_filter(signal, median_filter, fs):
    """Apply median filter to signal.

    Parameters
    ----------
    signal : array
    median_filter : scalar
        length of median filter window (in seconds) for removing slow components.
    fs : scalar
        sampling frequency

    Returns
    -------
    array
        filtered signal

    """

    win_size = int(median_filter * fs)
    if win_size % 2 == 0:
        win_size = win_size + 1
    series = pandas.Series(signal).rolling(window=win_size, center=True, min_periods=1)
    signal_filtered = signal - series.median()

    return signal_filtered


def apply_low_pass_filter(signal, cutoff, **kwargs):
    """Apply low-pass filter to signal.

    Parameters
    ----------
    signal : array
    band : scalar
        cut-off frequency for low-pass filter.
    axis : scalar, optional
        axis along which to filter.
    fs : scalar
        sampling frequency
    transition_width : str or scalar
        size of teransition between stop and pass bands
    attenuation: scalar
        stop-band attenuation in dB

    Returns
    -------
    array
        filtered signal

    """
    return apply_filter(signal, [None, float(cutoff)], **kwargs)


def apply_high_pass_filter(signal, cutoff, **kwargs):
    """Apply high-pass filter to signal.

    Parameters
    ----------
    signal : array
    band : scalar
        cut-off frequency for high-pass filter.
    axis : scalar, optional
        axis along which to filter.
    fs : scalar
        sampling frequency
    transition_width : str or scalar
        size of teransition between stop and pass bands
    attenuation: scalar
        stop-band attenuation in dB

    Returns
    -------
    array
        filtered signal

    """
    return apply_filter(signal, [float(cutoff), None], **kwargs)


def inspect_filter(
    b, a=1.0, fs=1.0, npoints=None, filtfilt=False, detail=None, grid=False
):
    """Plot filter characteristics.

    Parameters
    ----------
    b : filter coefficients (numerator)
    a : filter coefficients (denominator), optional
    fs : scalar
        sampling frequency
    npoints : scalar
        number of points to plot
    filtfilt : bool
        if True, will plot the filter's amplitude response assuming
        forward/backward filtering scheme. If False, will plot the
        filter's amplitude and phase responses and the group delay.
    detail : 2-element sequence
        plot an additional zoomed in digital filter response with the
        given frequency bounds.
    grid : bool
        if True, all plots will have both axes grids turned on.

    """
    # prepare plot
    fig = plt.figure()

    ncols, ratios = (1, [1]) if detail is None else (2, [2, 1])
    nrows = 3 if not filtfilt else 1

    g = gridspec.GridSpec(
        nrows=nrows,
        ncols=ncols,
        bottom=0.1,
        top=0.9,
        left=0.1,
        right=0.9,
        wspace=0.5,
        hspace=0.5,
        width_ratios=ratios,
    )

    host = mpl_toolkits.axes_grid1.host_subplot(g[0, 0])
    plot_filter_amplitude(
        b, a, fs, npoints=npoints, filtfilt=filtfilt, axes=host, grid=grid
    )

    if not filtfilt:
        ax_phase = mpl_toolkits.axes_grid1.host_subplot(g[1, 0], sharex=host)
        plot_filter_phase(b, a, fs, npoints=npoints, axes=ax_phase, grid=grid)

        ax_delay = mpl_toolkits.axes_grid1.host_subplot(g[2, 0], sharex=host)
        plot_filter_group_delay(b, a, fs, npoints=npoints, axes=ax_delay, grid=grid)

    if detail is not None:
        host_detail = mpl_toolkits.axes_grid1.host_subplot(g[0, 1], sharey=host)
        plot_filter_amplitude(
            b,
            a,
            fs,
            npoints=npoints,
            filtfilt=filtfilt,
            freq_lim=detail,
            axes=host_detail,
            grid=grid,
        )

        if not filtfilt:
            ax = mpl_toolkits.axes_grid1.host_subplot(
                g[1, 1], sharex=host_detail, sharey=ax_phase
            )
            plot_filter_phase(
                b, a, fs, npoints=npoints, freq_lim=detail, axes=ax, grid=grid
            )

            ax = mpl_toolkits.axes_grid1.host_subplot(
                g[2, 1], sharex=host_detail, sharey=ax_delay
            )
            plot_filter_group_delay(
                b, a, fs, npoints=npoints, freq_lim=detail, axes=ax, grid=grid
            )


def plot_filter_amplitude(
    b, a=1.0, fs=1.0, npoints=None, freq_lim=None, filtfilt=False, axes=None, grid=False
):
    """Plot filter amplitude characteristics.

    Parameters
    ----------
    b, a : scalar or 1d array
        Filter coefficients.
    fs : scalar
        Sampling frequency.
    freq_lim : (min, max)
        Frequency limits for plotting.
    filtfilt : bool
        Perform bidirectional filtering using filtfilt
    axes : matplotlib axes
    grid : bool
        Display grid.

    Returns
    -------
    axes

    """
    if axes is None:
        fig = plt.figure()
        axes = mpl_toolkits.axes_grid1.host_subplot(111)

    w, h = scipy.signal.freqz(b, a, worN=npoints)

    freq = 0.5 * w * fs / np.pi
    h = np.abs(h)
    if filtfilt:
        h = h ** 2

    axes.plot(freq, 20 * np.log10(h), "k")
    plt.setp(axes, xlabel="Frequency [Hz]", ylabel="Amplitude [dB]")
    axes.grid(grid)

    par = axes.twinx()
    par.plot(freq, h, "b")
    par.set_ylabel("Normalized amplitude")
    par.yaxis.get_label().set_color("b")
    par.grid(False)

    if freq_lim is not None:
        axes.set_xlim(freq_lim)

    return axes


def plot_filter_phase(
    b, a=1.0, fs=1.0, npoints=None, freq_lim=None, axes=None, grid=False
):
    """Plot filter phase characteristics.

    Parameters
    ----------
    b, a : scalar or 1d array
        Filter coefficients.
    fs : scalar
        Sampling frequency.
    freq_lim : (min, max)
        Frequency limits for plotting.
    axes : matplotlib axes
    grid : bool
        Display grid.

    Returns
    -------
    axes

    """
    if axes is None:
        fig = plt.figure()
        axes = mpl_toolkits.axes_grid1.host_subplot(111)

    w, h = scipy.signal.freqz(b, a, worN=npoints)

    freq = 0.5 * w * fs / np.pi
    phase = np.unwrap(np.angle(h))

    axes.plot(freq, phase, "k")
    plt.setp(axes, ylabel="Phase [radians]", xlabel="Frequency [Hz]")
    axes.grid(grid)

    if freq_lim is not None:
        axes.set_xlim(freq_lim)

    return axes


def plot_filter_group_delay(
    b, a=1.0, fs=1.0, npoints=None, freq_lim=None, axes=None, grid=False
):
    """Plot filter group delay characteristics.

    Parameters
    ----------
    b, a : scalar or 1d array
        Filter coefficients.
    fs : scalar
        Sampling frequency.
    freq_lim : (min, max)
        Frequency limits for plotting.
    axes : matplotlib axes
    grid : bool
        Display grid.

    Returns
    -------
    axes

    """
    if axes is None:
        fig = plt.figure()
        axes = mpl_toolkits.axes_grid1.host_subplot(111)

    w, delay = scipy.signal.group_delay((b, a), w=npoints)

    freq = 0.5 * w * fs / np.pi
    axes.plot(freq, delay, "k")
    plt.setp(axes, ylabel="Group delay [samples]", xlabel="Frequency [Hz]")
    axes.set_ylim(0, np.max(delay) * 1.2)
    axes.grid(grid)

    t = matplotlib.transforms.Affine2D()
    t.scale(1.0, fs / 1000.0)
    twin = axes.twin(t)
    twin.set_ylabel("Group delay [ms]")
    twin.xaxis.set_visible(False)
    twin.grid(False)

    if freq_lim is not None:
        axes.set_xlim(freq_lim)

    return axes


def compute_envelope(
    signals,
    freq_band=None,
    axis=-1,
    fs=1.0,
    isfiltered=False,
    median_filter=None,
    filter_options={},
    smooth_options={},
    pad=True,
):
    """Compute average envelope of band-pass filtered signal.

    Parameters
    ----------
    signals : array
        either array with raw signals (isfiltered==False) or
        pre-filtered signals (isfiltered==True). Can also be a sequence
        of such signals.
    freq_band : str or 2-element sequence, optional
        frequency band (in case signal needs to filtered)
    axis : scalar, optional
        axis of the time dimension in the signals array
    fs : scalar, optional
        sampling frequency
    median_filter : None or scalar, optional
        length of median filter window (in seconds) for removing slow components
    isfiltered : bool, optional
    filter_options : dict, optional
        dictionary with options for filtering (if signal is not already filtered).
    smooth_options : dict, optional
        dictionary with optional kernel and bandwidth keys for envelope
        smoothing
    pad : bool, optional
        allow zero-padding of signal to nearest power of 2 or 3 in order
        to speed up computation

    See Also
    --------
    apply_filter
    construct_filter
    apply_median_filter
    fklab.signals.smooth.smooth1d

    Returns
    -------
    envelope : 1D array

    """
    # filter
    if not isfiltered:
        if freq_band is None:
            raise ValueError("Please specify frequency band")
        filter_arg = dict(transition_width="25%", attenuation=60)
        filter_arg.update(filter_options)

        envelope = apply_filter(signals, freq_band, fs=fs, axis=axis, **filter_arg)
    else:
        envelope = signals

    # compute envelope
    if not isinstance(envelope, (tuple, list)):
        envelope = [envelope]

    if len(envelope) == 0:
        raise ValueError("No signal provided.")

    # check that all arrays in the list have the same size along axis
    if not all([x.shape[axis] == envelope[0].shape[axis] for x in envelope]):
        raise ValueError("Signals in list do not have compatible shapes")

    N = envelope[0].shape[axis]
    if pad:
        Norig = N
        N = int(np.min([2, 3] ** np.ceil(np.log(N) / np.log([2, 3]))))

    envelope = envelope.copy()  # to avoid altering input list

    for k in range(len(envelope)):
        _envelope = envelope[k]
        _envelope = np.abs(scipy.signal.hilbert(_envelope, N=N, axis=axis))

        if _envelope.ndim > 1:
            s = _envelope.shape[axis]
            _envelope = np.mean(np.rollaxis(_envelope, axis).reshape((s, -1)), axis=1)

        envelope[k] = _envelope

    if len(envelope) > 1:
        envelope = reduce(np.add, envelope) / len(envelope)
    else:
        envelope = envelope[0]

    if pad:
        envelope = envelope[:Norig]

    if median_filter:
        envelope = apply_median_filter(envelope, median_filter, fs)

    # (optional) smooth envelope
    smooth_arg = dict(kernel="gaussian", bandwidth=-1.0)
    smooth_arg.update(smooth_options)
    if smooth_arg["bandwidth"] > 0:
        envelope = fklab.signals.smooth.smooth1d(envelope, delta=1.0 / fs, **smooth_arg)

    return envelope


def compute_sliding_rms(
    signals,
    freq_band=None,
    axis=-1,
    fs=1.0,
    isfiltered=False,
    filter_options={},
    smooth_options={},
):
    """Compute root-mean-square of band-pass filtered signal.

    Parameters
    ----------
    signals : array
        either array with raw signals (isfiltered==False) or
        pre-filtered signals (isfiltered==True). Can also be a sequence
        of such signals.
    freq_band : str or 2-element sequence, optional
        frequency band (in case signal needs to filtered)
    axis : scalar, optional
        axis of the time dimension in the signals array
    fs : scalar, optional
        sampling frequency
    isfiltered : bool, optional
    filter_options : dict, optional
        dictionary with options for filtering (if signal is not already filtered).
    smooth_options : dict, optional
        dictionary with optional kernel and bandwidth keys for smoothing

    See Also
    --------
    apply_filter
    construct_filter
    fklab.signals.smooth.smooth1d

    Returns
    -------
    rms : 1D array
        root-mean-square of band-pass filtered signal. If multiple signals
        are provided, these are averaged after squaring. Time-weighted
        averaging (i.e. smoothing) is also performed on the squared signal.

    """
    # filter
    if not isfiltered:
        if freq_band is None:
            raise ValueError("Please specify frequency band")
        filter_arg = dict(transition_width="25%", attenuation=60)
        filter_arg.update(filter_options)
        envelope = apply_filter(signals, freq_band, axis=axis, fs=fs, **filter_arg)
    else:
        envelope = signals

    # compute envelope
    if not isinstance(envelope, (tuple, list)):
        envelope = [envelope]

    if len(envelope) == 0:
        raise ValueError("No signal provided.")

    # check that all arrays in the list have the same size along axis
    if not all([x.shape[axis] == envelope[0].shape[axis] for x in envelope]):
        raise ValueError("Signals in list do not have compatible shapes")

    envelope = envelope.copy()  # to avoid altering input list

    for k in range(len(envelope)):
        envelope[k] = envelope[k] ** 2
        if envelope[k].ndim > 1:
            envelope[k] = np.mean(
                np.rollaxis(envelope[k], axis).reshape(
                    [
                        envelope[k].shape[axis],
                        envelope[k].size / envelope[k].shape[axis],
                    ]
                ),
                axis=1,
            )

    if len(envelope) > 1:
        envelope = reduce(np.add, envelope) / len(envelope)
    else:
        envelope = envelope[0]

    smooth_arg = dict(kernel="gaussian", bandwidth=-1.0)
    smooth_arg.update(smooth_options)
    if smooth_arg["bandwidth"] > 0:
        envelope = fklab.signals.smooth.smooth1d(
            envelope, axis=axis, delta=1.0 / fs, **smooth_arg
        )

    envelope = np.sqrt(envelope)

    return envelope


def contrast_frequency_bands(
    signal,
    fs=2000.0,
    target=[160, 225],
    contrast=[[100, 140], [250, 400]],
    weights=None,
    kind="power",
    transition_width="10%",
    smooth=0.01,
):
    """Compute difference between target and contrast frequency bands.

    Parameters
    ----------
    signal : 1d or 2d array, or list of 1d or 2d arrays
    fs : float
        Sampling frequency
    target : 1d-array like with shape (2,)
    contrast : 2d array-like with shape (n,2)
    weights : None or 1d array like with shape (n,)
    kind: str
        One of: `power`, `power x frequency` or `envelope`.
    transition_width : float or str
        Transition width for band-pass filters.
    smooth : float
        Kernel bandwidth for smoothing of power or envelope

    Returns
    -------
    contrast : 1d array

    """

    if not kind in ("power", "power x frequency", "envelope"):
        raise ValueError("Unknown value for kind argument.")

    all_bands = np.row_stack([target, contrast])

    if not isinstance(signal, (list, tuple)):
        signal = [signal]

    signal = [y[:, None] if y.ndim == 1 else y for y in signal]

    # filter signal
    y = [
        np.dstack(
            [
                apply_filter(x, band, fs=fs, axis=0, transition_width=transition_width)
                for band in all_bands
            ]
        )
        for x in signal
    ]

    if kind == "power":
        y = [k ** 2 for k in y]
    elif kind == "power x frequency":
        fcenter = np.array([(a + b) / 2 for a, b in all_bands])[None, None, :]
        y = [(k ** 2) * fcenter for k in y]
    else:  # kind == "envelope"
        y = [np.abs(scipy.signal.hilbert(k, axis=0)) for k in y]

    if weights is None:
        weights = np.ones(len(contrast)) / len(contrast)

    y = [k[:, :, 0] - np.average(k[:, :, 1:], axis=2, weights=weights) for k in y]

    if not smooth is None:
        y = [
            fklab.signals.smooth.smooth1d(k, axis=0, delta=1.0 / fs, bandwidth=smooth)
            for k in y
        ]

    y = np.mean(np.concatenate(y, axis=1), axis=1)

    return y
