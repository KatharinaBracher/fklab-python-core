"""
===============================================================
Multi-rate functions (:mod:`fklab.signals.multirate.multirate`)
===============================================================

.. currentmodule:: fklab.signals.multirate.multirate

Functions to increase or decrease a signal's sampling rate.


"""
from fklab.version._core_version._version import __version__

__all__ = [
    "upsample",
    "interp",
    "interp_time_vector",
    "factorization",
    "decimate",
    "resample",
]

import numpy as np
import scipy as sp
import scipy.signal
from scipy.signal import decimate as _decimate
from scipy.signal import resample


def upsample(x, factor, axis=-1):
    """Upsample signal by inserting zeros.

    Parameters
    ----------
    x : ndarray
    factor : int
        Upscaling factor.
    axis : int
        Array axis along which to upscale.

    Returns
    -------
    ndarray

    """
    x = np.asarray(x)

    shape = list(x.shape)
    ndim = len(shape)

    factor = int(factor)

    if factor == 1:
        return x
    elif factor < 1:
        raise ValueError

    shape[axis] = shape[axis] * factor

    # create output array
    y = np.zeros(shape, dtype=x.dtype)

    indices = [slice(None)] * ndim
    indices[axis] = slice(None, None, factor)

    y[indices] = x

    return y


def interp(x, factor, axis=-1, L=4, alpha=0.5, window="blackman"):
    """Increase sampling rate by factor using interpolation filter.

    Parameters
    ----------
    x : ndarray
    factor : int
        Upsampling factor.
    axis : int
        Array axis along which to upsample.
    L : int
        Filter length, calculated as 2*L*factor+1.
    alpha : float
        Normalized cut-off frequency for filter.
    window : str
        Filter window.

    Returns
    -------
    ndarray

    """
    # upsample data
    y = upsample(x, factor, axis=axis)

    # create low pass filter
    filter_length = 2 * L * factor + 1
    F, M = (
        [0.0, 2.0 * alpha / factor, 2.0 * alpha / factor, 1.0],
        [factor, factor, 0.0, 0.0],
    )  # frequency and magnitude specification
    b = sp.signal.firwin2(
        filter_length,
        F,
        M,
        nfreqs=2 ** (np.ceil(np.log2(filter_length)) + 2) + 1,
        window=window,
    )
    a = 1.0

    # to minimize edge effects, data at begin (end) of array is rotated 180 degrees
    # around first (last) point
    shape = list(x.shape)
    shape[axis] = len(b) - 1
    zi = np.zeros(shape)

    # mirror/reflect left edge, upsample and filter
    pre = 2 * np.take(x, [0], axis=axis) - np.take(
        x, np.arange(2 * L + 1, 0, -1), axis=axis
    )
    pre = upsample(pre, factor, axis=axis)

    pre, zi = sp.signal.lfilter(b, a, pre, axis=axis, zi=zi)

    # filter main data
    data, zi = sp.signal.lfilter(b, a, y, axis=axis, zi=zi)
    data = np.roll(data, -L * factor, axis=axis)

    # mirror/reflect right edge, upsample and filter
    post = 2 * np.take(x, [-1], axis=axis) - np.take(
        x, np.arange(-2, -2 * L - 2, -1), axis=axis
    )
    post = upsample(post, factor, axis=axis)

    post, zi = sp.signal.lfilter(b, a, post, axis=axis, zi=zi)

    indices = [slice(None)] * len(shape)
    indices[axis] = slice(-L * factor, None)
    data[indices] = np.take(post, np.arange(L * factor), axis=axis)

    return data


def interp_time_vector(t, dt, factor):
    """Interpolate time vector.

    Parameters
    ----------
    t : 1d array
        Time vector.
    dt : scalar
        Interval between time values.
    factor : int
        Upsampling factor

    Returns
    -------
    1d array

    """
    ts = np.arange(factor).reshape(1, factor) * dt / factor
    ts = ts + t.reshape(t.size, 1)
    ts = ts.flatten()
    return ts


def factorization(v: int, x: int):
    """Factorize integer `v` into factors smaller than `x`

    Parameters
    ----------
    v : int
        Value to factorize.
    x : int
        Maximum factor.

    Returns
    -------
    factors : tuple of ints

    """

    ret = []
    while v > x:
        for i in range(x, 1, -1):
            if v % i == 0:
                ret.append(i)
                v //= i
                break
        else:
            raise RuntimeError(
                f"Value {v} cannot be factorized to factors smaller than {x}"
            )

    ret.append(v)
    return tuple(ret)


def decimate(x, q, **kwargs):
    """Downsample the signal after applying an anti-aliasing filter.

    This is a thin wrapper around scipy.signal.decimate that checks if
    the (integer) downsampling factor is larger than 12, in which case
    the downsampling factor is facorized to multiple downsampling factors
    smaller than 12 (if possible) and decimation is performed multiple
    times. Alternatively, it is possible to directly specify an iterable
    of downsample factors.

    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR
    filter with Hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    x : array_like
        The signal to be downsampled, as an N-dimensional array.
    q : int or iterable of int
        The downsampling factor(s). If the downsampling factor
        is larger than 12, then it is factorized into smaller factors
        and decimation is performed multiple times in sequence.
        If factorization is not possible, then an exception is raised.
    n : int, optional
        The order of the filter (1 less than the length for 'fir'). Defaults to
        8 for 'iir' and 20 times the downsampling factor for 'fir'.
    ftype : str {'iir', 'fir'} or ``dlti`` instance, optional
        If 'iir' or 'fir', specifies the type of lowpass filter. If an instance
        of an `dlti` object, uses that object to filter before downsampling.
    axis : int, optional
        The axis along which to decimate.
    zero_phase : bool, optional
        Prevent phase shift by filtering with `filtfilt` instead of `lfilter`
        when using an IIR filter, and shifting the outputs back by the filter's
        group delay when using an FIR filter. The default value of ``True`` is
        recommended, since a phase shift is generally not desired.

    Returns
    -------
    y : ndarray
        The down-sampled signal.

    """

    if isinstance(q, int):
        if q == 1:
            return x
        q = factorization(q, 12)
    else:
        q = np.asarray(q, dtype=int).ravel()

    for factor in q:
        x = _decimate(x, factor, **kwargs)

    return x
