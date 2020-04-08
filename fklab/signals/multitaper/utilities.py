"""
Utilities
=========
"""
import math

import numpy as np
import scipy as sp

__all__ = ["nextpow2", "mtfft", "mtoptions"]


def nextpow2(n):
    """
    Compute the first power of 2 larger than a given value.

    Parameters
    ----------
    n : number

    Returns
    -------
    exponent : integer
        The smallest integer such that 2\*\*exponent is larger than n

    """
    n = np.abs(n)
    val, p = 1, 0
    while val < n:
        val = val * 2
        p += 1
    return p


def _compute_nfft(n, pad=0):
    """Compute padded number of samples for FFT.

    Parameters
    ----------
    n : int
        number of samples to pad
    pad : int, optional
        amount of padding to a higher power of 2. Zero means no padding.

    Returns
    -------
    n : int
        padded number of samples

    """
    return max(2 ** (nextpow2(n + 1) + pad - 1), n)


def _allowable_bandwidth(n, fs=1.0):
    """Multitape bandwidth limits.

    Parameters
    ----------
    n : int
        Number of samples in signal
    fs : float, optional
        Sampling frequency of signal

    Returns
    -------
    [min, max]
        minimum and maximum allowable bandwidths

    """
    return [fs / n, 0.5 * fs * (n - 1) / n]


# @functools.lru_cache(maxsize=2)
def _check_bandwidth(bw, n, fs=1.0, correct=False):
    """Check and correct multitaper bandwidth.

    Parameters
    ----------
    bw : float
        Requested bandwidth
    n : int
        Number of samples in signal
    fs : float, optional
        Sampling frequency
    correct : bool, optional
        Correct bandwidth if requested value is out of range.

    Returns
    -------
    bw : float
        Multitaper bandwidth

    """
    if bw is None:
        TW = min(3, (n - 1) / 2.0)
        bw = TW * fs / n
    else:
        bw = float(bw)
        limits = _allowable_bandwidth(n, fs)

        if bw < limits[0] or bw > limits[1]:
            if correct:
                bw = max(min(limits[1], bw), limits[0])
            else:
                raise ValueError(
                    "Bandwidth out of range. Minimum bandwidth = {minbw} Hz. Maximum bandwidth = {maxbw} Hz.".format(
                        minbw=limits[0], maxbw=limits[-1]
                    )
                )

    return bw


# @functools.lru_cache(maxsize=2)
def _check_ntapers(ntapers, n, bw=None, fs=1.0, correct=False):
    """Check and correct number of tapers.

    Parameters
    ----------
    ntapers : int
        Requested number of tapers
    n : int
        Number of samples in signal
    bw : float, optional
        Requested bandwidth
    fs : float, optional
        Sampling frequency
    correct : bool
        Correct number of tapers and bandwidth if requested values are
        out of range.

    Returns
    -------
    ntapers : int
        Number of tapers
    bw : float
        Multitaper bandwidth

    """
    bw = _check_bandwidth(bw, n, fs, correct)

    TW = bw * n / fs
    maxtapers = int(math.floor(2 * TW - 1))

    if ntapers is None:
        ntapers = maxtapers
    else:
        ntapers = int(ntapers)
        if ntapers < 1 or ntapers > maxtapers:
            if correct:
                ntapers = max(min(maxtapers, ntapers), 1)
            else:
                raise ValueError(
                    "Invalid number of tapers. Maximum number of tapers = {maxtapers}".format(
                        maxtapers=maxtapers
                    )
                )

    return ntapers, bw


# @functools.lru_cache(maxsize=2)
def _compute_tapers(bw, n, fs, ntapers, correct=False):
    """Compute tapers.

    Parameters
    ----------
    bw : float
        Requested bandwidth
    n : int
        Number of samples in signal
    fs : float
        Sampling frequency
    ntapers : int
        Requested number of tapers
    correct : bool
        Correct number of tapers and bandwidth if requested values are
        out of range.

    Returns
    -------
    tapers : ndarray

    """
    ntapers, bw = _check_ntapers(ntapers, n, bw, fs, correct)
    # tapers = spectrum.dpss( int(n), int( bw*n/fs ), int(ntapers) )[0] * np.sqrt(fs)
    tapers = sp.signal.windows.dpss(int(n), int(bw * n / fs), int(ntapers)) * np.sqrt(
        fs
    )
    tapers = tapers.T

    return tapers


class mtoptions(object):
    """Class to manage multitaper options.

    Parameters
    ----------
    bandwidth : scalar
    fpass : scalar or [min, max]
    error : {'none', 'theory', 'jackknife'}
    pvalue : scalar
    pad : int
    ntapers : int

    """

    def __init__(
        self, bandwidth=None, fpass=None, error="none", pvalue=0.05, pad=0, ntapers=None
    ):
        self.bandwidth = bandwidth
        self.fpass = fpass
        self.error = error
        self.pvalue = pvalue
        self.pad = pad
        self.ntapers = ntapers

    def keys(self):
        return ["bandwidth", "fpass", "error", "pvalue", "pad", "ntapers"]

    def __getitem__(self, key):
        if key in list(self.keys()):
            return object.__getattribute__(self, key)
        else:
            raise KeyError("Unknown key")

    def bandwidth_range(self, nsamples, fs=1.0):
        """Permissible range of bandwidths.

        Parameters
        ----------
        nsamples : int
            Number of samples in signal
        fs : float, optional
            Sampling frequency of signal

        Returns
        -------
        [min, max]
            minimum and maximum allowable bandwidths

        """
        return _allowable_bandwidth(nsamples, fs)

    def nfft(self, nsamples):
        """Compute padded number of samples for FFT.

        Parameters
        ----------
        n : int
            number of samples to pad

        Returns
        -------
        n : int
            padded number of samples

        """
        return _compute_nfft(nsamples, self._pad)

    def frequencies(self, nsamples, fs=1.0):
        """Compute frequency vector.

        Parameters
        ----------
        nsamples : int
            Number of samples
        fs : scalar
            Sampling frequency

        Returns
        -------
        f : 1d array
            Frequencies
        fidx : 1d array
            Indices of selected frequencies that fall within the fpass
            setting.

        """
        nfft = self.nfft(nsamples)
        f = fs * np.arange(nfft) / nfft

        if self._fpass is None:
            fpass = [0.0, fs / 2.0]
        else:
            fpass = self._fpass

        fidx = np.logical_and(f >= fpass[0], f <= fpass[1])

        return f, fidx

    def validate(self, nsamples, fs=1.0, correct=False):
        """Validate multitaper options.

        Parameters
        ----------
        nsamples : int
            Number of samples
        fs : scalar
            Sampling frequency
        correct : bool
            Correct bandwidth and tapers if needed.

        Returns
        -------
        dict
            Validated multitaper options and pre-computed tapers.

        """
        nsamples = int(nsamples)
        if nsamples < 3:
            raise ValueError("Number of samples should be at least 3.")

        fs = float(fs)
        if fs <= 0.0:
            raise ValueError("Sampling frequency should be larger than zero.")

        d = dict(
            sampling_frequency=fs,
            nsamples=nsamples,
            error=self._error,
            pvalue=self._pvalue,
            pad=self.pad,
            nfft=self.nfft(nsamples),
        )

        d["ntapers"], d["bandwidth"] = _check_ntapers(
            self._ntapers, nsamples, bw=self._bw, fs=fs, correct=correct
        )
        d["frequencies"], d["fpass"] = self.frequencies(nsamples, fs)

        d["tapers"] = self.tapers(nsamples, fs, correct)

        return d

    def tapers(self, nsamples, fs=1.0, correct=False):
        """Compute tapers.

        Parameters
        ----------
        nsamples : int
            Number of samples in signal
        fs : float
            Sampling frequency
        correct : bool
            Correct number of tapers and bandwidth if requested values are
            out of range.

        Returns
        -------
        tapers : ndarray

        """
        return _compute_tapers(self._bw, nsamples, fs, self._ntapers, correct)

    @property
    def bandwidth(self):
        """Bandwidth for tapers."""
        return self._bw

    @bandwidth.setter
    def bandwidth(self, val):
        if not val is None:
            val = float(val)
            if val <= 0.0:
                raise ValueError()
        self._bw = val

    @property
    def fpass(self):
        """Select frequency band of interest."""
        return self._fpass

    @fpass.setter
    def fpass(self, val):
        if not val is None:
            val = list(np.array(val, dtype=np.float64).ravel())
            if len(val) == 1:
                val = [0.0, val[0]]
            elif not len(val) == 2:
                raise ValueError("FPass should be a scalar or 2-element sequence.")

            if val[0] < 0.0 or val[0] > val[1]:
                raise ValueError(
                    "FPass values should be larger than zero and strictly monotonically increasing."
                )

        self._fpass = val

    @property
    def error(self):
        """Type of error to compute ('none', 'theory', 'jackknife')."""
        return self._error

    @error.setter
    def error(self, val):
        if val is None or not val:
            val = "none"
        elif not val in ("none", "theory", "jackknife"):
            raise ValueError("Error should be one of 'none', 'theory', 'jackknife'.")
        self._error = val

    @property
    def pvalue(self):
        """P-value for error computation."""
        return self._pvalue

    @pvalue.setter
    def pvalue(self, val):
        val = float(val)
        if val <= 0.0 or val >= 1:
            raise ValueError("p-Value should be between zero and one.")

        self._pvalue = val

    @property
    def pad(self):
        """Select amount of padding."""
        return self._pad

    @pad.setter
    def pad(self, val):
        if not val is None:
            val = int(val)
            if val < 0:
                raise ValueError("Pad should be equal to or larger than zero.")

        self._pad = val

    @property
    def ntapers(self):
        """Get number of tapers."""
        return self._ntapers

    @ntapers.setter
    def ntapers(self, val):
        if not val is None:
            val = int(val)
            if val < 1:
                raise ValueError("Number of tapers should be larger than zero.")

        self._ntapers = val


def mtfft(data, tapers, nfft, fs):
    """Define multi-tapered FFT.

    Parameters
    ----------
    data : 2d array
        data array with samples along first axis and signals along the second axis
    tapers: 2d array
        tapers with samples along first axis and tapers along the second axis
    nfft : integer
        number of points for FFT calculation
    fs : float
        sampling frequency of data

    Returns
    -------
    J : 3d array
        FFT of tapered signals. Shape of the array is (samples, tapers, signals)

    """
    # TODO: check shape of m and tapers

    data = np.array(data)
    tapers = np.array(tapers)

    nfft = int(nfft)
    fs = float(fs)

    data_proj = data[:, None, :] * tapers[:, :, None]

    J = np.fft.fft(data_proj, nfft, axis=0) / fs

    return J
