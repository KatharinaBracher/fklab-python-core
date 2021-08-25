"""
==========================================================================
Multi-taper spectral analysis (:mod:`fklab.signals.multitaper.multitaper`)
==========================================================================

.. currentmodule:: fklab.signals.multitaper.multitaper

Functions for multi-taper spectral analysis, including spectrum,
spectrogram, coherence and coherogram. The multitaper implementation is
based on the open-source Chronux Matlab toolbox (http://www.chronux.org).

Spectral analysis
=================

"""
import numpy as np
import scipy as sp
import scipy.stats

from .utilities import mtfft
from .utilities import mtoptions
from .utilities import nextpow2
from fklab.signals.core import extract_data_windows
from fklab.signals.core import extract_trigger_windows
from fklab.signals.core import generate_windows


__all__ = ["mtspectrum", "mtspectrogram", "mtcoherence", "mtcoherogram"]


def _spectrum_error(S, J, errtype, pval, avg, numsp=None):
    if errtype == "none" or errtype is None:
        return None

    nf, K, C = J.shape  # freq x tapers x channels

    if S.ndim == 1:
        S = S[:, None]

    if numsp is not None:
        numsp = np.array(numsp, copy=False).ravel()
        if len(numsp) != C:
            raise ValueError("Invalid value for numsp")

    pp = 1.0 - float(pval) / 2.0
    qq = 1.0 - pp

    if avg:
        dim = K * C
        C = 1
        dof = 2 * dim * np.ones(1)  # degrees of freedom

        if not numsp is None:
            dof = np.fix(1.0 / (1.0 / dof + 1.0 / (2.0 * np.sum(numsp))))

        J = np.reshape(J, (nf, dim, C))

    else:

        dim = K
        dof = 2.0 * dim * np.ones(C)

        if not numsp is None:
            for ch in range(C):
                dof[ch] = np.fix(1.0 / (1.0 / dof + 1.0 / (2.0 * numsp[ch])))

    Serr = np.zeros((2, nf, C))

    if errtype == "theory":
        Qp = sp.stats.chi2.ppf(pp, dof)
        Qq = sp.stats.chi2.ppf(qq, dof)

        # check size of dof and Qp, Qq
        # either could be scalar of vector

        Serr[0, :, :] = dof[None, :] * S / Qp[None, :]
        Serr[1, :, :] = dof[None, :] * S / Qq[None, :]

    else:  # errtype == 'jackknife'
        tcrit = sp.stats.t.ppf(pp, dim - 1)

        Sjk = np.zeros((dim, nf, C))

        for k in range(dim):
            Jjk = J[:, np.setdiff1d(list(range(dim)), [k]), :]  # 1-drop projection
            eJjk = np.sum(Jjk * np.conjugate(Jjk), axis=1)
            Sjk[k, :, :] = eJjk / (dim - 1)  # 1-drop spectrum

        sigma = np.sqrt(dim - 1) * np.std(np.log(Sjk), axis=0)

        # if C==1; sigma=sigma'; end
        #
        # conf=repmat(tcrit,nf,C).*sigma;
        # conf=squeeze(conf);

        conf = tcrit * sigma
        Serr[0, :, :] = S * np.exp(-conf)
        Serr[1, :, :] = S * np.exp(conf)

    # Serr=shiftdim( squeeze(Serr), 1 );

    if avg:
        # drop last axis for average case
        return Serr[:, :, 0]
    else:
        return Serr


def _mtspectrum_single(data, fs=1.0, average=False, **kwargs):
    """Compute multi-tapered spectrum of vectors.

    Parameters
    ----------
    data : 2d array
        data array with samples along first axis and channels along the second axis
    fs : float, optional
        sampling frequency
    average : bool
        compute averaged spectrum
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. pad=0 means no
        padding, pad=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers

    Returns
    -------
    S : vector or 2d array
        spectral density, with shape (frequencies, signals) or
        (frequencies,) if average==True.
    f : 1d array
        vector of frequencies
    Serr : None, 2d array or 3d array
        lower and upper error estimates. The shape of the array is
        (2, frequencies, signals), or (2, frequencies) if average==True,
        where the first axis contains the lower and upper error estimates.
    options : dict

    """
    # TODO: check data
    data = np.array(data, copy=False)
    N = data.shape[0]

    options = mtoptions(**kwargs)
    options = options.validate(N, fs)

    f = options["frequencies"][options["fpass"]]

    J = mtfft(data, options["tapers"], options["nfft"], options["sampling_frequency"])
    J = J[options["fpass"]]
    S = 2 * np.mean(
        np.real(np.conjugate(J) * J), axis=1
    )  # factor two because we combine positive and negative frequencies

    if average:
        S = np.mean(S, axis=1)

    Serr = _spectrum_error(S, J, options["error"], options["pvalue"], average)

    return S, f, Serr, options


def mtspectrum(
    data,
    fs=1.0,
    start_time=0.0,
    window_size=None,
    epochs=None,
    average=True,
    triggers=None,
    **kwargs
):
    """Compute windowed multi-tapered spectrum.

    Parameters
    ----------
    data : 1d array
        data vector
    fs : float, optional
        sampling frequency
    starttime : float, optional
        time of the first sample in seconds
    window_size : float, optional
        window size in seconds
    epochs : segment like, optional
        data epochs for which to compute spectrum
    average : bool, optional
        compute averaged spectrum
    triggers : 1d array, optional
        Times of triggers for a triggered spectrum
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. pad=0 means no
        padding, pad=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers


    Returns
    -------
    S : vector or 2d array
        spectral density, with shape (frequencies, signals) or
        (frequencies,) if average==True.
    f : 1d array
        vector of frequencies
    Serr : None or 3d array
        lower and upper error estimates. The shape of the array is
        (2, frequencies, signals), or (2, frequencies) if average==True,
        where the first axis contains the lower and upper error estimates.
    options : dict

    """
    # check data
    data = np.array(data, copy=False)
    if data.ndim != 1:
        raise ValueError("Only vector data is supported")

    if triggers is None:
        _, data = extract_data_windows(
            data, window_size, start_time=start_time, fs=fs, epochs=epochs
        )
    else:
        _, data = extract_trigger_windows(
            data, triggers, window_size, start_time=start_time, fs=fs, epochs=epochs
        )

    return _mtspectrum_single(data, fs=fs, average=average, **kwargs)


def mtspectrogram(
    data,
    fs=1.0,
    start_time=0.0,
    window_size=None,
    window_overlap=0.0,
    epochs=None,
    average=True,
    triggers=None,
    trigger_window=1.0,
    **kwargs
):
    """Compute multi-tapered spectrogram.

    Parameters
    ----------
    data : 1d array
        data vector
    fs : scalar, optional
        sampling frequency
    start_time : float, optional
        time of the first sample in seconds
    window_size : float, optional
        window size in seconds
    window_overlap : float, optional
        Fraction of overlap between neighbouring windows
    epochs : segment like, optional
        data epochs for which to compute spectrum
    average : bool
        compute averaged spectrum
    triggers : 1d array, optional
        Times of triggers for a triggered spectrum
    trigger_window : float or [left, right]
        Window around trigger for which to compute spectrogram
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. pad=0 means no
        padding, pad=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers

    Returns
    -------
    S : 2d array or 3d array
        spectral density, with shape (time, frequencies, signals) or
        (time, frequencies) if average==True.
    t : 1d array
        vector of times
    f : 1d array
        vector of frequencies
    Serr : None, 3d array or 4d array
        lower and upper error estimates. The shape of the array is
        (time, 2, frequencies, signals), or (time, 2, frequencies)
        if average==True, where the second axis contains the
        lower and upper error estimates
    options : dict

    """
    data = np.array(data, copy=False)

    if triggers is None:
        if data.ndim == 1:
            data = np.reshape(data, (len(data), 1))
        elif data.ndim != 2:
            raise ValueError("Data should be vector or 2d array.")
    elif data.ndim != 1:
        raise ValueError("For triggered spectrogram, data should be a vector.")

    if not triggers is None:
        _, data = extract_trigger_windows(
            data, triggers, trigger_window, start_time=start_time, fs=fs, epochs=epochs
        )

    n, ch = data.shape

    if triggers is None:
        s = start_time
    else:
        ## hack: copy of code from _generate_trigger_windows
        trigger_window = np.array(trigger_window, dtype=np.float, copy=False).ravel()
        if len(trigger_window) == 1:
            trigger_window = np.abs(trigger_window) * [-1, 1]
        elif len(trigger_window) != 2 or np.diff(trigger_window) <= 0.0:
            raise ValueError("Invalid window")
        s = trigger_window[0]

    nwin, t, idx = generate_windows(
        n, window_size, window_overlap=window_overlap, fs=fs, start_time=s, center=True
    )

    options = mtoptions(**kwargs)
    options = options.validate(np.round(np.float(window_size) * fs), fs)

    fpass = options["fpass"]
    f = options["frequencies"][fpass]
    numfreq = np.sum(fpass)

    Serr = None
    compute_error = options["error"] not in [None, "none"]

    if average:
        S = np.zeros((nwin, numfreq))
        if compute_error:
            Serr = np.zeros((nwin, 2, numfreq))
    else:
        S = np.zeros((nwin, numfreq, ch))
        if compute_error:
            Serr = np.zeros((nwin, 2, numfreq, ch))

    for k, indices in enumerate(idx()):
        # idx = np.arange( winstart[k], winstart[k] + window_size, dtype=np.int )
        J = mtfft(
            data[indices, :],
            options["tapers"],
            options["nfft"],
            options["sampling_frequency"],
        )
        J = J[fpass, :]

        if average:
            # factor two because we combine positive and negative frequencies
            S[k] = 2 * np.mean(np.mean(np.real(np.conjugate(J) * J), axis=1), axis=1)
        else:
            # factor two because we combine positive and negative frequencies
            S[k] = 2 * np.mean(np.real(np.conjugate(J) * J), axis=1)

        if compute_error:
            Serr[k] = _spectrum_error(
                S[k], J, options["error"], options["pvalue"], average
            )

    return S, t, f, Serr, options


def _coherence_error(c, j1, j2, errtype, pval, avg, numsp1=None, numsp2=None):
    if errtype == "none" or errtype is None:
        return None, None, None

    nf, K, Ch = j1.shape  # freq x tapers x channels

    if c.ndim == 1:
        c = c[:, None]

    if numsp1 is not None:
        numsp1 = np.array(numsp1, copy=False).ravel()
        if len(numsp1) != Ch:
            raise ValueError("Invalid value for numsp1")

    if numsp2 is not None:
        numsp2 = np.array(numsp2, copy=False).ravel()
        if len(numsp2) != Ch:
            raise ValueError("Invalid value for numsp2")

    pp = 1 - float(pval) / 2.0

    # find the number of degress of freedom
    if avg:
        dim = K * Ch

        if not numsp1 is None:
            dof1 = np.fix(2 * np.sum(numsp1) * 2 * dim / (2 * np.sum(numsp1) + 2 * dim))
        else:
            dof1 = 2 * dim

        if not numsp2 is None:
            dof2 = np.fix(2 * np.sum(numsp2) * 2 * dim / (2 * np.sum(numsp2) + 2 * dim))
        else:
            dof2 = 2 * dim

        dof = np.array(min(dof1, dof2), copy=False).ravel()
        Ch = 1

        j1 = j1.reshape((nf, dim, 1))
        j2 = j2.reshape((nf, dim, 1))
    else:
        dim = K
        dof = 2 * dim

        if not numsp1 is None:
            dof1 = np.fix(2 * numsp1 * 2 * dim / (2 * numsp1 + 2 * dim))
        else:
            dof1 = 2 * dim

        if not numsp2 is None:
            dof2 = np.fix(2 * numsp2 * 2 * dim / (2 * numsp2 + 2 * dim))
        else:
            dof2 = 2 * dim

        dof = np.minimum(dof1, dof2)

    # theoretical, asymptotic confidence level
    df = np.ones(dof.shape, dtype=np.float)
    df[dof > 2] = 1.0 / ((dof / 2.0) - 1)
    confC = np.sqrt(1.0 - pval ** df)

    # phase standard deviation (theoretical and jackknife) and
    # jackknife confidence intervals for c
    if errtype == "theory":
        phistd = np.sqrt((2.0 / dof[None, :] * (1.0 / (c ** 2) - 1)))
        Cerr = None
    else:  # errtype=='jackknife'
        tcrit = sp.stats.t.ppf(pp, dof - 1)
        atanhCxyk = np.empty((dim, nf, Ch))
        phasefactorxyk = np.empty((dim, nf, Ch), dtype=np.complex)
        Cerr = np.empty((2, nf, Ch))
        for k in range(dim):  # dim is the number of 'independent estimates'
            indxk = np.setdiff1d(list(range(dim)), k)
            j1k = j1[:, indxk, :]
            j2k = j2[:, indxk, :]
            ej1k = np.sum(j1k * np.conjugate(j1k), 1)
            ej2k = np.sum(j2k * np.conjugate(j2k), 1)
            ej12k = np.sum(j1k * np.conjugate(j2k), 1)
            Cxyk = ej12k / np.sqrt(ej1k * ej2k)
            absCxyk = np.abs(Cxyk)
            atanhCxyk[k] = np.sqrt(2 * dim - 1) * np.arctanh(
                absCxyk
            )  # 1-drop estimate of z
            phasefactorxyk[k] = Cxyk / absCxyk

        atanhC = np.sqrt(2 * dim - 2) * np.arctanh(c)
        sigma12 = np.sqrt(dim - 1) * np.std(
            atanhCxyk, axis=0
        )  # jackknife estimate std(z)=sqrt(dim-1)*std of 1-drop estimate

        Cu = atanhC + tcrit[None, :] * sigma12
        Cl = atanhC - tcrit[None, :] * sigma12

        Cerr[0] = np.maximum(np.tanh(Cl / np.sqrt(2 * dim)), 0.0)
        Cerr[1] = np.tanh(Cu / np.sqrt(2 * dim - 2))

        phistd = np.sqrt((2 * dim - 2) * (1 - np.abs(np.mean(phasefactorxyk, axis=0))))

    return confC, phistd, Cerr


def _mtcoherence_single(x, y, fs=1.0, average=False, **kwargs):
    """Compute multi-tapered coherence of two vectors.

    Parameters
    ----------
    x : 2d array
    y : 2d array
    fs : float, optional
        Sampling frequency
    average : bool
        compute averaged spectrum
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. pad=0 means no
        padding, pad=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers

    Returns
    -------
    c : ndarray
        Coherence
    phi : ndarray
        Phase
    f : 1d array
        vector of frequencies
    err : 3-element tuple
        with the following items: 1) theoretical, asymptotic confidence
        level, 2) phase standard deviation and 3) coherence confidence
        interval.
    options : dict

    """
    # TODO: check shape of x and y

    x = np.array(x, copy=False)
    y = np.array(y, copy=False)

    N = x.shape[0]

    options = mtoptions(**kwargs)
    options = options.validate(N, fs)

    f = options["frequencies"][options["fpass"]]

    J1 = mtfft(x, options["tapers"], options["nfft"], options["sampling_frequency"])
    J2 = mtfft(y, options["tapers"], options["nfft"], options["sampling_frequency"])
    J1 = J1[options["fpass"]]
    J2 = J2[options["fpass"]]

    S12 = np.mean(np.conjugate(J1) * J2, axis=1)
    S1 = np.mean(np.conjugate(J1) * J1, axis=1)
    S2 = np.mean(np.conjugate(J2) * J2, axis=1)

    if average:
        S12 = np.mean(S12, axis=1)
        S1 = np.mean(S1, axis=1)
        S2 = np.mean(S2, axis=1)

    C12 = S12 / np.sqrt(S1 * S2)
    c = np.abs(C12)
    phi = np.angle(C12)

    err = _coherence_error(c, J1, J2, options["error"], options["pvalue"], average)

    return c, phi, f, err, options


def mtcoherence(
    x,
    y,
    fs=1.0,
    start_time=0,
    window_size=None,
    epochs=None,
    average=True,
    triggers=None,
    **kwargs
):
    """Compute windowed multi-tapered coherence of two vectors.

    Parameters
    ----------
    x : 1d array
    y : 1d array
    fs : float, optional
        Sampling frequency
    start_time : float
       time of the first sample in seconds
    window_size : float, optional
        window size in seconds
    epochs : segment like, optional
        data epochs for which to compute spectrum
    average : bool
        compute averaged spectrum
    triggers : 1d array, optional
        Times of triggers for a triggered spectrum
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. pad=0 means no
        padding, pad=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers

    Returns
    -------
    c : ndarray
        Coherence
    phi : ndarray
        Phase
    f : 1d array
        vector of frequencies
    err : 3-element tuple
        with the following items: 1) theoretical, asymptotic confidence
        level, 2) phase standard deviation and 3) coherence confidence
        interval.
    options : dict

    """
    # check data
    x = np.array(x, copy=False)
    y = np.array(y, copy=False)

    if x.ndim != 1:
        raise ValueError("Only vector data is supported")

    if triggers is None:
        _, x = extract_data_windows(
            x, window_size, start_time=start_time, fs=fs, epochs=epochs
        )
        _, y = extract_data_windows(
            y, window_size, start_time=start_time, fs=fs, epochs=epochs
        )
    else:
        _, x = extract_trigger_windows(
            x, triggers, window_size, start_time=start_time, fs=fs, epochs=epochs
        )
        _, y = extract_trigger_windows(
            y, triggers, window_size, start_time=start_time, fs=fs, epochs=epochs
        )

    return _mtcoherence_single(x, y, fs=fs, average=average, **kwargs)


def mtcoherogram(
    x,
    y,
    fs=1.0,
    start_time=0.0,
    window_size=None,
    window_overlap=0.0,
    epochs=None,
    average=True,
    triggers=None,
    trigger_window=1.0,
    **kwargs
):
    """Compute multi-tapered coherogram.

    Parameters
    ----------
    x : 1d array
    y : 1d array
    fs : float, optional
        Sampling frequency
    start_time : float
       time of the first sample in seconds
    window_size : float, optional
        window size in seconds
    window_overlap : float, optional
        Fraction of overlap between neighbouring windows
    epochs : segment like, optional
        data epochs for which to compute spectrum
    average : bool
        compute averaged spectrum
    triggers : 1d array, optional
        Times of triggers for a triggered spectrum
    trigger_window : float or [left, right]
        Window around trigger for which to compute spectrogram
    bandwidth : float, optional
        Spectral bandwidth
    fpass : float or [float, float], optional
        Range of frequencies that should be retained, either specified
        as a single upper bound or a (lower, upper) range.
    error : {'none', 'theory', 'jackknife'}, optional
        Kind of error to estimate
    pvalue : float, optional
        P-value for the error estimate
    pad : int, optional
        Amount of signal padding for fft computation. pad=0 means no
        padding, pad=1 means pad to the next power of 2 larger than
        the length of the signal, etc.
    ntapers : int, optional
        Desired number of tapers

    Returns
    -------
    c : ndarray
        Coherence
    phi : ndarray
        Phase
    t : 1d array
        vector of times
    f : 1d array
        vector of frequencies
    err : 3-element tuple
        with the following items: 1) theoretical, asymptotic confidence
        level, 2) phase standard deviation and 3) coherence confidence
        interval. Note: currently error calculations are not supported
        for coherograms
    options : dict

    """
    x = np.array(x, copy=False)
    y = np.array(y, copy=False)

    if x.shape != y.shape:
        raise ValueError("Input signals need to have the same size.")

    if triggers is None:
        if x.ndim == 1:
            x = np.reshape(x, (len(x), 1))
            y = np.reshape(y, (len(y), 1))
        elif x.ndim != 2:
            raise ValueError("Data should be vector or 2d array.")
    elif x.ndim != 1:
        raise ValueError("For triggered spectrogram, data should be a vector.")

    if not triggers is None:
        _, x = extract_trigger_windows(
            x, triggers, trigger_window, start_time=start_time, fs=fs, epochs=epochs
        )
        _, y = extract_trigger_windows(
            y, triggers, trigger_window, start_time=start_time, fs=fs, epochs=epochs
        )

    n, ch = x.shape

    if triggers is None:
        s = start_time
    else:
        ## hack: copy of code from _generate_trigger_windows
        trigger_window = np.array(trigger_window, dtype=np.float, copy=False).ravel()
        if len(trigger_window) == 1:
            trigger_window = np.abs(trigger_window) * [-1, 1]
        elif len(trigger_window) != 2 or np.diff(trigger_window) <= 0.0:
            raise ValueError("Invalid window")
        s = trigger_window[0]

    nwin, t, idx = generate_windows(
        n, window_size, window_overlap=window_overlap, fs=fs, start_time=s, center=True
    )

    options = mtoptions(**kwargs)
    options = options.validate(np.round(np.float(window_size) * fs), fs)

    fpass = options["fpass"]
    f = options["frequencies"][fpass]
    numfreq = np.sum(fpass)

    Serr = None
    compute_error = options["error"] not in [None, "none"]

    if average:
        coh = np.zeros((nwin, numfreq))
        phi = np.zeros((nwin, numfreq))
        # if compute_error:
        #    Serr = np.zeros( (nwin,numfreq,2) )
    else:
        coh = np.zeros((nwin, numfreq, ch))
        phi = np.zeros((nwin, numfreq, ch))
        # if compute_error:
        #    Serr = np.zeros( (nwin,numfreq,ch,2) )

    for k, indices in enumerate(idx()):

        J1 = mtfft(
            x[indices, :],
            options["tapers"],
            options["nfft"],
            options["sampling_frequency"],
        )
        J2 = mtfft(
            y[indices, :],
            options["tapers"],
            options["nfft"],
            options["sampling_frequency"],
        )
        J1 = J1[fpass, :]
        J2 = J2[fpass, :]

        S12 = np.mean(np.conjugate(J1) * J2, axis=1)
        S1 = np.mean(np.conjugate(J1) * J1, axis=1)
        S2 = np.mean(np.conjugate(J2) * J2, axis=1)

        if average:
            S12 = np.mean(S12, axis=1)
            S1 = np.mean(S1, axis=1)
            S2 = np.mean(S2, axis=1)

        C12 = S12 / np.sqrt(S1 * S2)
        coh[k] = np.abs(C12)
        phi[k] = np.angle(C12)

        # if compute_error:
        #    pass

    return coh, phi, t, f, (None, None, None), options
