"""
=======================================================
Event algorithms (:mod:`fklab.events.basic_algorithms`)
=======================================================

.. currentmodule:: fklab.events.basic_algorithms

Provides basic algorithms for event time vectors.
"""
import math
import warnings

import numba
import numpy as np
import scipy as sp
import scipy.interpolate

import fklab.segments
import fklab.utilities.general
from fklab.segments.basic_algorithms import check_segments
from fklab.segments.basic_algorithms import segment_contains
from fklab.segments.basic_algorithms import segment_intersection
from fklab.segments.basic_algorithms import segment_remove_overlap
from fklab.version._core_version._version import __version__

__all__ = [
    "split_eventstrings",
    "event_rate",
    "event_bin",
    "event_bursts",
    "filter_bursts",
    "filter_bursts_length",
    "event_count",
    "event_intervals",
    "filter_intervals",
    "complex_spike_index",
    "event_correlation",
    "joint_event_correlation",
    "peri_event_histogram",
    "peri_event_density",
    "check_events",
    "check_events_list",
    "fastbin",
    "spike_time_tiling_coefficient",
    "shift_events",
]


def split_eventstrings(timestamp, eventstrings):
    """Split event strings.

    Converts a sequence of timestamps and corresponding event strings to
    a dictionary with for each unique event string, the timestamps at
    which the event happened.

    Parameters
    ----------
    timestamp : 1d array-like
    eventstrings : sequence of str

    Returns
    -------
    dict

    """

    timestamp = np.asarray(timestamp)
    events = np.unique(eventstrings)
    d = {e: timestamp[np.array(eventstrings) == e] for e in events}
    return d


def check_events(x, copy=True):
    """Convert input to event vector.

    Parameters
    ----------
    x : array-like
    copy : bool
        the output vector will always be a copy of the input

    Returns
    -------
    1d array

    """
    return fklab.utilities.general.check_vector(x, copy=copy, real=True)


def check_events_list(x, copy=True):
    """Convert input to sequence of event vectors.

    Parameters
    ----------
    x : array-like or sequence of array-likes
    copy : bool
        the output vectors will always be copies of the inputs

    Returns
    -------
    tuple of 1d arrays

    """
    return fklab.utilities.general.check_vector_list(x, copy=copy, real=True)


def event_rate(events, segments=None, separate=False):
    """Return mean rate of events.

    Parameters
    ----------
    events : 1d numpy array or sequence of 1d numpy arrays
        vector(s) of event times (in seconds)
    segments : (n,2) array or Segment, optional
        array of time segment start and end times ([start, stop[)
    separate : bool, optional
        compute event rates for all segments separately

    Returns
    -------
    rate : array
        Mean firing rate for each of the input event time vectors.  If separate=True, then a 2d array is returned, where
        rate[i,j] represents the mean firing rate for event vector i and segment j

    """
    events = check_events_list(events)

    if segments is None:
        return np.array(
            [
                event_rate(
                    x,
                    segments=[
                        np.min(x) - 0.5 * np.mean(np.diff(x)),
                        np.max(x) + 0.5 * np.mean(np.diff(x)),
                    ],
                )
                for x in events
            ]
        ).flatten()

    segments = check_segments(segments, copy=False)
    if not separate:
        # combine segments
        segments = segment_remove_overlap(segments)
        # find number of events in segments
        ne = [np.sum(segment_contains(segments, x)[1]) for x in events]
        ne = np.float64(ne)
        # convert to rate
        fr = ne / np.sum(np.diff(segments, axis=1))
    else:
        # find number of events in each segment
        ne = [segment_contains(segments, x)[1] for x in events]
        ne = np.float64(np.vstack(ne))
        # convert to rate
        fr = ne / np.diff(segments, axis=1).reshape((1, len(segments)))

    return fr


def event_bin(events, bins, kind="count"):
    """Count number of events in bins.

    Parameters
    ----------
    events : 1d array or sequence of 1d arrays
        vector(s) of sorted event times (in seconds)
    bins : ndarray
        array of time bin start and end times (in seconds). Can be either
        a vector of sorted times, or a (n,2) array of bin start and
        end times.
    kind : {'count','binary','rate'}, optional
        determines what count to return for each bin. This can be the
        number of events (count), the presence or absence of events
        (binary) or the local event rate in the time bin (rate).

    Returns
    -------
    counts : 1d array
        event counts for each bin

    """
    events = check_events_list(events)

    bins = check_segments(bins)

    sortflag = False
    if not fklab.utilities.general.isascending(bins[:, 0]):
        sort_idx = np.argsort(bins[:, 0], axis=0, kind="mergesort")
        bins = bins[sort_idx]
        sortflag = True

    m = np.zeros((bins.shape[0], len(events)), dtype=np.uint64)

    # for each event vector, compute histogram
    # and sort event vector if needed (will be slower)
    # note that histc cannot be used since it does not support overlapping
    # bins, something that fastbin does support
    for k, e in enumerate(events):
        if fklab.utilities.general.isascending(e):
            m[:, k] = fastbin(e, bins.astype(float))
        else:
            m[:, k] = fastbin(np.sort(e), bins.astype(float))

    # transpose output, such that rows represent events and columns represent bins
    # m = m.T

    if kind == "count":
        pass
    elif kind == "binary":
        m[m > 0] = 1
        m = np.uint8(m)
    elif kind == "rate":
        m = m / np.diff(bins, axis=1)
    else:
        raise NotImplementedError("kind can be only count, binary or rate.")

    # if bins had to be sorted initially, unsort them here
    if sortflag:
        m = m[np.argsort(sort_idx)]

    return m


def event_bursts(events, intervals=None, nevents=None, marks=None):
    """Detect bursts of events.

    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    intervals : 2-element sequence, optional
        minimum and maximum inter-event time intervals to consider two
        events as part of a burst
    nevents : 2-element sequence, optional
        minimum and maximum number of events in a burst
    marks : 1d array, optional
        vector of event marks (e.g. spike amplitude)

    Returns
    -------
    1d array
        vector with burst indicators: 0=non-burst event, 1=first event
        in burst, 2=event in burst, 3=last event in burst

    """
    events = check_events(events, copy=False)
    n = len(events)

    if n == 0:
        return np.array([])

    if intervals is None:
        intervals = [0.003, 0.015]
    if nevents is None:
        nevents = [2, np.Inf]

    max_interval = intervals[-1]
    min_interval = intervals[0]

    if marks is not None:
        if len(marks) != n:
            raise ValueError("marks needs to have the same size than events")

    dpre = np.abs(event_intervals(events, kind="pre")[0])
    dpost = np.abs(event_intervals(events, kind="post")[0])

    inburst = np.zeros(n)

    if marks is not None:
        apre = event_intervals(marks, kind="pre")[0]
        apost = event_intervals(marks, kind="post")[0]
    else:
        apre = 0
        apost = 0

    # find first event in burst
    mask = np.logical_or.reduce(
        (dpre > max_interval, dpre < min_interval, np.isnan(dpre))
    )
    mask = np.logical_or(mask, apre < 0)
    mask = np.logical_and.reduce((mask, dpost <= max_interval, dpost >= min_interval))
    mask = np.logical_and(mask, apost <= 0)
    inburst[mask] = 1

    # find middle event(s) in burst
    mask = np.logical_and.reduce(
        (
            dpre <= max_interval,
            dpre >= min_interval,
            dpost <= max_interval,
            dpost >= min_interval,
        )
    )
    mask = np.logical_and(mask, np.logical_and(apre >= 0, apost <= 0))
    inburst[mask] = 2

    # find last event in burst
    mask = np.logical_or.reduce(
        (dpost > max_interval, dpost < min_interval, np.isnan(dpost))
    )
    mask = np.logical_and.reduce((mask, dpre <= max_interval, dpre >= min_interval))
    mask = np.logical_and(mask, apre >= 0)
    inburst[mask] = 3

    burst_start = np.flatnonzero(inburst == 1)
    burst_end = np.flatnonzero(inburst == 3)

    if len(burst_start) == 0 or len(burst_end) == 0:
        return inburst

    # find and remove incomplete bursts at start and end
    if burst_start[0] > burst_end[0]:
        inburst[burst_end[0]] = 0
        burst_end = np.delete(burst_end, 0)

    if burst_end[-1] < burst_start[-1]:
        inburst[burst_start[-1]] = 0
        burst_start = np.delete(burst_start, -1)

    # determine number of events in every burst
    ne = burst_end - burst_start + 1
    invalidbursts = np.logical_or(ne < nevents[0], ne > nevents[-1])

    # to get rid of invalid burst for now do a loop, until we've figured out a
    # better way of doing it
    for bs, be, b in zip(burst_start, burst_end, invalidbursts):
        if b:
            inburst[bs : (be + 1)] = 0

    return inburst


def filter_bursts(events, bursts=None, method="none", **kwargs):
    """Filter events based on participation in bursts.

    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    bursts : 1d array, optional
        burst indicator vector as returned by event_bursts function.
        If not provided, it will be computed internally (parameters to
        the event_bursts function can be provided as extra keyword arguments)
    method : {'none', 'reduce', 'remove', 'isolate', 'isolatereduce'}
        filter method to be applied. 'none': keep all events,
        'reduce': only keep non-burst events and first event in bursts,
        'remove': remove all burst events, 'isolate': remove all non-burst
        events, 'isolatereduce': only keep first event in bursts.

    Returns
    -------
    1d array
        filtered vector of event times
    1d bool array
        selection filter

    """
    if bursts is None:
        bursts = event_bursts(events, **kwargs)

    if method == "reduce":
        idx = bursts <= 1
    elif method == "remove":
        idx = bursts == 0
    elif method == "isolate":
        idx = bursts != 0
    elif method == "isolatereduce":
        idx = bursts == 1
    else:
        idx = bursts >= 0

    events = events[idx]

    return events, idx


def filter_bursts_length(bursts, nevents=None):
    """Filter bursts on number of events.

    Parameters
    ----------
    bursts : 1d array
        burst indicator vector as returned by event_bursts function.
    nevents : scalar or 2-element sequence
        range of burst lengths that will be filtered out. If `nevents` is a scalar, the range is [nevents, Inf]. By default, the number is 2 events minimum (= No filtering)

    Returns
    -------
    bursts : 1d array
        filtered burst indicator vector

    """
    if nevents is None:
        nevents = 2

    # find first and last events in bursts
    burst_start = np.flatnonzero(bursts == 1)
    burst_end = np.flatnonzero(bursts == 3)

    # determine burst lengths
    burstlen = burst_end - burst_start + 1

    # find burst to remove
    nevents = np.array(nevents).ravel()
    if len(nevents) > 1:
        burstremove = np.flatnonzero(
            np.logical_or(burstlen < nevents[0], burstlen > nevents[-1])
        )
    else:
        burstremove = np.flatnonzero(burstlen >= nevents[0])

    # loop to do actual removal
    for k in burstremove:
        bursts[burst_start[k] : (burst_end[k] + 1)] = 0

    return bursts


def event_count(events, x):
    """Calculate the cumulative event count.

    Parameters
    ----------
    events : 1d array
        vector of event times (in seconds)
    x : 1d array
        times at which to evaluate cumulative event count

    Returns
    -------
    count : 1d array
        event counts
    """
    events = check_events(events, copy=False)
    ne = len(events)

    x = np.array(x).ravel()
    nx = len(x)

    # combine event and x vectors, label 0/1 and sort
    tmp = np.concatenate((events, x))
    q = np.concatenate((np.zeros(ne), np.ones(nx)))
    q = q[tmp.argsort(),]
    qi = np.nonzero(q)[0]

    # compute cumulative count
    cs = np.cumsum(q)
    c = qi - cs[qi] + 1

    return c


def event_intervals(events, other=None, kind="post", segments=None):
    """Return inter-event intervals.

    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    other : 1d array, optional
        vector of sorted event times (in seconds)
    kind : {'pre', '<', 'post', '>', 'smallest', 'largest'}
        type of interval to return. 'pre' or '<': interval to previous event,'post' or '>': interval to next event, 'smallest' or 'largest':
        smallest/largest of the intervals to the previous and next events.
    segments : (n,2) array or Segment, optional
        array of time segment start and end times

    Returns
    -------
    intervals : 1d array
        the requested interval for each event in the input vector events. Intervals to events in the past have a negative sign.
    index : 1d array
        index of the event to which the interval was determined


    """
    events = check_events(events, copy=False)
    n = len(events)

    if n == 0:
        return np.array([]), np.array([])

    if not kind in ["pre", "<", "post", ">", "smallest", "largest"]:
        raise NotImplementedError(
            str(kind)
            + " is not the right keyword for kind. Available keys are : pre , <, post, >, smallest or largest"
        )

    if other is None:  # auto intervals
        d = np.diff(events)
        ipre = np.concatenate(([np.nan], -d))
        idxpre = np.int64(np.concatenate(([-1], np.arange(n - 1))))
        ipost = np.concatenate((d, [np.nan]))
        idxpost = np.int64(np.concatenate((np.arange(n - 1) + 1, [-1])))

    else:  # cross intervals
        other = np.array(other).ravel()
        n2 = len(other)

        if n2 == 0:
            return np.zeros(n) + np.nan, np.zeros(n) + np.nan

        idxpre = np.asarray(
            np.floor(np.interp(events, other, np.arange(n2), left=-1)), dtype=np.int64
        )
        valids = np.flatnonzero(idxpre >= 0)
        ipre = np.zeros(n) + np.nan

        if len(valids) > 0:
            ipre[valids] = other[idxpre[valids]] - events[valids]
            ipre[valids[-1] + 1 :] = other[-1] - events[valids[-1] + 1 :]
            idxpre[valids[-1] + 1 :] = n2

        idxpost = np.asarray(
            np.ceil(np.interp(events, other, np.arange(n2), right=-1)), dtype=np.int64
        )
        valids = np.flatnonzero(idxpost >= 0)
        ipost = np.zeros(n) + np.nan

        if len(valids) > 0:
            ipost[valids] = other[idxpost[valids]] - events[valids]
            ipost[0 : valids[0]] = other[0] - events[0 : valids[0]]
            idxpost[0 : valids[0]] = 0

    if kind in ["pre", "<"]:
        ii, idx = ipre, idxpre
    elif kind in ["post", ">"]:
        ii, idx = ipost, idxpost
    elif kind == "smallest":
        ii = ipre
        tmp = np.flatnonzero(np.abs(ipost) <= np.abs(ipre))
        ii[tmp] = ipost[tmp]
        ii[np.logical_or(np.isnan(ipre), np.isnan(ipost))] = np.nan

        idx = idxpre
        idx[tmp] = idxpost[tmp]
        idx[np.logical_or(np.isnan(ipre), np.isnan(ipost))] = -1

    elif kind == "largest":
        ii = ipre
        tmp = np.flatnonzero(np.abs(ipost) > np.abs(ipre))
        ii[tmp] = ipost[tmp]
        ii[np.logical_or(np.isnan(ipre), np.isnan(ipost))] = np.nan

        idx = idxpre
        idx[tmp] = idxpost[tmp]
        idx[np.logical_or(np.isnan(ipre), np.isnan(ipost))] = -1

    if segments is not None:
        segments = check_segments(segments)

        valid = idx >= 0
        valid = (valid) & (segment_contains(segments, events)[0])

        if other is not None:
            valid[valid] &= segment_contains(segments, other[idx[valid]])[0]
        else:
            valid[valid] &= segment_contains(segments, events[idx[valid]])[0]

        ii[~valid] = np.nan
        idx[~valid] = -1

    return ii, idx


def filter_intervals(events, mininterval=0.003):
    """Filter out events based on interval to previous event.

    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    mininterval : scalar, optional

    Returns
    -------
    events : 1d array
        filtered vector of sorted event times (in seconds)
    index : 1d array
        index into original vector of all removed events

    """
    events = check_events(events, copy=True)
    d = np.diff(events)
    idx = np.flatnonzero(d < mininterval) + 1
    events = np.delete(events, idx)
    return events, idx


def complex_spike_index(spike_times, spike_amp=None, intervals=None):
    """Compute complex spike index.

    Parameters
    ----------
    spike_times : 1d array or sequence of 1d arrays
        vector(s) of sorted event times (in seconds)
    spike_amp : 1d array or sequence of 1d arrays
        vector(s) of spike amplitudes
    intervals : 2-element sequence
        minimum and maximum inter-spike time intervals to consider two
        spikes as part of a burst

    Returns
    -------
    2d array
        array of complex spike indices between all pairs of spike time
        vectors

    """
    if intervals is None:
        intervals = [0.003, 0.015]

    spike_times = check_events_list(spike_times, copy=False)
    nvectors = len(spike_times)

    if spike_amp is not None:
        spike_amp = fklab.utilities.general.check_vector_list(spike_amp, copy=False)
        if len(spike_amp) != len(spike_times):
            raise ValueError
        if not all([len(x) == len(y) for x, y in zip(spike_times, spike_amp)]):
            raise ValueError

    min_interval = intervals[0]
    max_interval = intervals[-1]

    c = np.zeros((nvectors, nvectors))

    for k in range(nvectors):
        for j in range(nvectors):
            # find smallest ISI and corresponding delta amplitude for each spike
            if k == j:
                dt, idx = event_intervals(spike_times[k], kind="smallest")
            else:
                dt, idx = event_intervals(
                    spike_times[k], other=spike_times[j], kind="smallest"
                )

            ii = idx >= 0
            dt = -dt[ii]
            if spike_amp is None:
                da = 0
            else:
                da = spike_amp[k][ii] - spike_amp[j][idx[ii]]

            c[k, j] = _calc_csi(dt, da, max_interval, min_interval) // len(
                spike_times[k]
            )

    return c


def _calc_csi(dt, da, max_int, min_int):
    # find all valid intervals (i.e. interval smaller than or equal to max_int)
    valid = np.abs(dt) <= max_int

    # find intervals within refractory period
    refract = np.abs(dt) < min_int

    # find intervals for all quadrants
    q1 = np.logical_and(da <= 0, dt > 0)  # post intervals with smaller amplitude
    q2 = np.logical_and(da > 0, dt < 0)  # pre intervals with larger amplitude
    q3 = np.logical_and(da <= 0, dt < 0)  # pre intervals with smaller amplitude
    q4 = np.logical_and(da > 0, dt > 0)  # post intervals with larger amplitude

    # count the number of intervals that contribute positively to CSI
    # i.e. preceding intervals with da>0 and following intervals with da<0
    # (complex burst) which are both valid and not in the refractory priod
    pos = np.sum(np.logical_and.reduce((np.logical_or(q1, q2), valid, ~refract)))

    # count the number of intervals that contribute negatively to CSI
    neg = np.sum(np.logical_and(np.logical_or.reduce((q3, q4, refract)), valid))

    # calculate csi
    c = 100 * (pos - neg)

    return c


@numba.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def _bsearchi(vector, key):  # pragma: no cover
    nmemb = len(vector)

    left = 0
    right = nmemb - 1

    while left <= right:
        mid = int(math.floor((left + right) / 2))

        if vector[mid] == key:
            return mid

        if vector[mid] > key:
            right = mid - 1
        else:
            left = mid + 1

    if (left > (nmemb - 1)) or (right < 0):
        return -1
    else:
        return right + (key - vector[right]) / (vector[left] - vector[right])


@numba.jit("uint64[:](f8[:],f8[:,:])", nopython=True, nogil=True)
def fastbin(events, bins):  # pragma: no cover
    """Count number of events in bins.

    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    bins : (n,2) array
        array of time bin start and end times (in seconds). The bins
        need to be sorted by start time.

    Returns
    -------
    counts : 1d array
        event counts for each bin

    """
    nevents = events.shape[0]
    nbins = bins.shape[0]

    counts = np.zeros((nbins,), dtype=np.uint64)

    next_idx = 0

    for k in range(nbins):
        while next_idx < nevents and (
            events[next_idx] < bins[k, 0] or np.isnan(events[next_idx])
        ):
            next_idx += 1

        idx = next_idx

        while idx < nevents and (events[idx] < bins[k, 1] or np.isnan(events[idx])):
            if ~np.isnan(events[idx]):
                counts[k] += 1
            idx += 1

    return counts


@numba.jit("u8(f8[:],f8[:],f8,f8,f8[:,:],i8,i8[:,:])", nopython=True, nogil=True)
def _find_events_near_reference(
    ref, ev, minlag, maxlag, segs, mode, out
):  # pragma: no cover
    # mode = 0: biased
    #   include all references and events inside segments
    # mode = 1: unbiased, strict
    #   include all events, but include only references for which
    #   surrounding [minlag, maxlag] window falls completely
    #   inside segments
    # mode = 2: unbiased, relaxed
    #   include all references, inside segments and include all
    #   events within [minlag, maxlag] window of references, even
    #   if events are outside segments

    nref = len(ref)  # number of reference events
    nev = len(ev)  # number of events
    nseg = len(segs)  # number of segments

    event_i = 0  # index of first event to be processed next

    n = 0  # number of valid reference events (inside segments)

    # loop through all segments
    for k in range(nseg):
        # find index i1 of first reference event within segment
        if segs[k, 0] < ref[0]:
            i1 = 0
        else:
            tmp = segs[k, 0]

            if mode == 1:
                # adjust segment boundary
                tmp = tmp - minlag

            if tmp > ref[-1]:
                continue

            i1 = int(math.ceil(_bsearchi(ref, tmp)))

        # find index i2 of last reference event within segment
        if segs[k, 1] > ref[nref - 1]:
            i2 = nref - 1
        else:
            tmp = segs[k, 1]

            if mode == 1:
                # adjust segment boundary
                tmp = tmp - maxlag

            if tmp < ref[0]:
                continue

            i2 = int(math.floor(_bsearchi(ref, tmp)))

        if i1 > i2 or i1 < 0 or i2 < 0:
            continue

        n += i2 - i1 + 1

        # loop through all reference events in segment
        for l in range(i1, i2 + 1):
            i = event_i
            event_i_set = 0
            while (
                i < nev
                and ev[i] <= ref[l] + maxlag
                and (mode == 2 or ev[i] <= segs[k, 1])
            ):
                if ev[i] >= ref[l] + minlag and (mode == 2 or ev[i] >= segs[k, 0]):
                    if event_i_set == 0:
                        out[l, 0] = i
                        event_i_set = 1
                        event_i = i
                i += 1

            if event_i_set:
                out[l, 1] = i - out[l, 0]

            # this code is slower
            # out[l,0] = math.ceil( bsearchi(ev,ref[l]+minlag) )
            # out[l,1] = math.floor( bsearchi(ev,ref[l]+maxlag) )

    return n


@numba.jit("f8[:](f8[:],f8[:],i8[:,:],f8[:])", nopython=True, nogil=True)
def _align_events(events, reference, idx, x):  # pragma: no cover
    n = len(reference)

    startidx = 0
    for k in range(n):
        if idx[k, 1] > 0:
            for j in range(idx[k, 1]):
                x[startidx + j] = events[idx[k, 0] + j] - reference[k]
            startidx = startidx + idx[k, 1]

    return x


def event_correlation(
    events,
    reference=None,
    lags=None,
    segments=None,
    remove_zero_lag=False,
    unbiased=False,
    return_index=False,
    return_time=False,
):
    """Find events surrounding reference events.

    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    reference : 1d array
        vector(s) of sorted reference event times (in seconds).
        If not provided, then events are used as a reference.
    lags : (float, float)
        minimum and maximum lag.
    segments : (n,2) array or Segment, optional
        array of time segment start and end times
    remove_zero_lag : bool, optional
        remove zero lag events
    unbiased : bool or one of {'none', 'strict', 'relaxed'}, optional
        If True or 'strict', only include reference events for which data
        is available at all lags. If 'relaxed', then all references inside
        segments are considered and events at all lags are counted even
        if they may lie outside segments.
    return_index : bool
        additionally, return indices of events near reference events
    return_time : bool or str
        additionally, return times of events near reference events.
        If return_time is 'reference', returns the time of the reference
        event instead.

    Returns
    -------
    rel_event_time : array
        time differences between events and reference events
    nvalid : int
        number of valid reference events
    index : (n,2) array, optional
        for each reference event, the start and end indices of neighboring
        events
    time : (n,) array, optional
        for each event near a reference events, the time of the event
        or the time of the corresponding reference event.

    """
    unbiased = {False: 0, True: 1, "none": 0, "strict": 1, "relaxed": 2}[unbiased]

    events = check_events(events, copy=False)

    if reference is None:
        reference = events
    else:
        reference = check_events(reference, copy=False)

    if lags is None:
        minlag = -1.0
        maxlag = 1.0
    else:
        minlag = float(lags[0])
        maxlag = float(lags[-1])

    if minlag > maxlag:
        raise ValueError("minimum lag should be smaller than maximum lag")

    if segments is None:
        segments = np.array([[-1, 1]], dtype=np.float64) * np.inf
    else:
        segments = check_segments(segments)

    if return_time == True:
        return_time = "event"
    elif return_time != False and not return_time in ("reference", "event"):
        raise ValueError(
            "Unknown value for return_time. Valid options are True, False, 'reference' and 'event'."
        )

    # remove overlap between segments
    segments = segment_remove_overlap(segments, strict=False)

    idx = np.zeros((len(reference), 2), dtype=np.int64)

    nvalidref = _find_events_near_reference(
        reference, events, minlag, maxlag, segments, unbiased, idx
    )

    n = np.sum(idx[:, 1])  # total number of events found near references

    x = np.zeros((n,), dtype=np.float64)

    x = _align_events(events, reference, idx, x)

    if remove_zero_lag:
        keep = x != 0
        x = x[keep]
    else:
        keep = np.full_like(x, True, dtype=bool)

    return_values = [x, nvalidref]

    if return_index:
        return_values.append(idx)

    if not return_time == False:
        if return_time == "event":
            t = [events[a : a + b] for a, b in idx]
        elif return_time == "reference":
            t = [
                np.full(b, reference[k], dtype=reference.dtype)
                for k, (a, b) in enumerate(idx)
            ]
        return_values.append(np.concatenate(t)[keep])

    return tuple(return_values)


def joint_event_correlation(
    events,
    reference=None,
    var_t=None,
    var=None,
    segments=None,
    minlag=-0.5,
    maxlag=0.5,
    remove_zero_lag=True,
    unbiased="relaxed",
    reference_time="reference",  # 'reference' or 'event'
    interp_options=None,
):
    """Jointly assess temporal and variable correlation of events.

    Parameters
    ----------
    events : 1d array
        vector of sorted event times (in seconds)
    reference : 1d array, optional
        vector of sorted reference event times (in seconds). If not provided,
        then events are used as a reference.
    var_t : 1d array
        time vector for variable
    var : array
        variable array, can be 2-dimensional, but first dimension must be time.
    segments : (n,2) array or Segment, optional
        array of time segment start and end times
    minlag, maxlag : float
        minimum and maximum lag times
    remove_zero_lag : bool
        remove events at zero time lag
    unbiased : bool or one of {'none', 'strict', 'relaxed'}
    reference_time : str, one of {'reference', 'event'}
        If 'reference', uses the time of the reference event to compute the
        associated variable value. If 'event', uses the actual event time.
    interp_options : None or dict
        Options for interpolation of variable at event times. By default,
        interpolation kind is 'nearest'

    Returns
    -------
    rel_event_time : 1d array
    event_var : array
    nvalid : int

    """

    interpolation_options = {"kind": "nearest"}
    if not interp_options is None:
        interpolation_options.update(interp_options)

    if var_t is None or var is None:
        raise ValueError("Please provide var_t and var arrays.")

    rel_event_time, nvalid, t = event_correlation(
        events,
        reference,
        lags=[minlag, maxlag],
        segments=segments,
        unbiased=unbiased,
        return_time=reference_time,
        remove_zero_lag=remove_zero_lag,
    )

    # interpolate variable for each neighboring spike
    event_var = scipy.interpolate.interp1d(var_t, var, **interpolation_options)(t)

    return rel_event_time, event_var, nvalid


def peri_event_histogram(
    events,
    reference=None,
    lags=None,
    segments=None,
    normalization="none",
    unbiased=False,
    remove_zero_lag=False,
):
    """Compute peri-event time histogram.

    Parameters
    ----------
    events : 1d array or sequence of 1d arrays
        vector(s) of sorted event times (in seconds)

    reference : 1d array or sequence of 1d arrays, optional
        vector(s) of sorted reference event times (in seconds). If not provided, then events are used as a reference.

    lags : 1d array, optional
        vector of sorted lag times that specify the histogram time bins

    segments : (n,2) array or Segment, optional
        array of time segment start and end times

    normalization : {'none', 'coef', 'rate', 'conditional mean intensity', 'product density', 'cross covariance', 'standard cross covariance', 'cumulant density', 'zscore'}, optional
        type of normalization

    unbiased : bool or {'none', 'stict', 'relaxed'}, optional
        If True or 'strict', only include reference events for which data
        is available at all lags. If 'relaxed', then all references inside
        segments are considered and events at all lags are counted even
        if they may lie outside segments.

    remove_zero_lag : bool, optional
        remove zero lag event counts

    Returns
    -------
    histogram: array with peri-event histogram, shape is (lags, events, references)
    bins: array with time lag bins used to compute histogram, shape is (nbins,2)


    .. note:: Developer's documentation of the various normalizations in peri-event histogram.

        T : total time
        mean rate:
        P0 = N0/T
        P1 = N1/T
        cross-correlation histogram: J10 = eventcorr( spike 0, spike 1 )
        cross product density: P10 = J/bT
        asymptotic distribution (independent):  sqrt(P10) -> sqrt(P1P0) +- norminv(1-alpha/2)/sqrt(4bT)
        conditional mean intensity: m10 = J/bN0
        asymptotic distribution (independent): qrt(m10) -> sqrt(P1) +- norminv(1-alpha/2)/sqrt(4bN0)
        cross-covariance / cumulant density: q10 = P10 - P1P0
        asymptotic distribution (independent):  q10 -> 0 +- norminv(1-alpha/2)*sqrt(P1P0/Tb)
        variance normalized q10 is approx normal only when normal approx to Poisson
        distribution applies, i.e. when lamda > 20 (see Lubenov&Siapas,2005)
        this gives condition bTP0P1>20
    """
    events = check_events_list(events, copy=False)

    if reference is None:
        reference = events
    else:
        reference = check_events_list(reference, copy=False)

    if lags is None:
        lags = np.linspace(-1, 1, 51)

    if segments is None:
        segments = np.array([[x[0], x[-1]] for x in events + reference])
        segments = np.array([[np.min(segments[:, 0]), np.max(segments[:, 1])]])

    segments = check_segments(segments, copy=False)

    duration = np.sum(np.diff(segments, axis=1))

    nev = len(events)
    nref = len(reference)
    nlags = len(lags) - 1
    minlag = lags[0]
    maxlag = lags[-1]

    p = np.zeros((nlags, nev, nref), dtype=np.float64)
    nvalid = np.zeros((nref,))

    for t in range(nref):
        for k in range(nev):
            tmp, nvalid[t] = event_correlation(
                events[k],
                reference[t],
                lags=[minlag, maxlag],
                segments=segments,
                unbiased=unbiased,
                remove_zero_lag=remove_zero_lag,
            )

            p[:, k, t] = np.histogram(tmp, bins=lags)[0]

    if (unbiased == True or unbiased == "strict") and normalization not in [
        "coef",
        "rate",
        "conditional mean intensity",
    ]:
        tmp = np.array([np.sum(segment_contains(segments, x)[0]) for x in reference])
        p = p * (tmp[None, None, :] / nvalid[None, None, :])

    if normalization in ["coef"]:
        p = p / nvalid[None, None, :]
    elif normalization in ["rate", "conditional mean intensity"]:
        p = p / (nvalid[None, None, :] * np.diff(lags)[:, None, None])
    elif normalization in ["product density"]:
        p = p / (np.diff(lags)[:, None, None] * duration)
    elif normalization in [
        "cross covariance",
        "cumulant density",
        "standard cross covariance",
    ]:
        refrate = np.array(
            [np.sum(segment_contains(segments, x)[0]) / duration for x in reference]
        )
        evrate = np.array(
            [np.sum(segment_contains(segments, x)[0]) / duration for x in events]
        )
        p = (
            p / (np.diff(lags)[:, None, None] * duration)
            - evrate[None, :, None] * refrate[None, None, :]
        )
        if normalization == "standard cross covariance":
            p *= np.sqrt(
                (np.diff(lags)[:, None, None] * duration)
                / (evrate[None, :, None] * refrate[None, None, :])
            )
    elif normalization in ["zscore"]:
        refrate = np.array(
            [np.sum(segment_contains(segments, x)[0]) / duration for x in reference]
        )
        evrate = np.array(
            [np.sum(segment_contains(segments, x)[0]) / duration for x in events]
        )
        p1p0 = evrate[None, :, None] * refrate[None, None, :]
        p = p / (np.diff(lags)[:, None, None] * duration) - p1p0
        p = p / (p1p0 / (np.diff(lags)[:, None, None] * duration))

    bins = np.vstack([lags[0:-1], lags[1:]]).T

    return p, bins


def peri_event_density(
    events,
    triggers,
    lags=None,
    npoints=101,
    bandwidth=0.05,
    squeeze=True,
    remove_zero_lag=False,
    segments=None,
    unbiased=True,
):
    """
    Compute peri-event probability distribution

    Parameters
    ----------
    events : 1d array or sequence of 1d arrays
        Vector(s) of sorted event times (in seconds)
    triggers : 1d array or sequence of 1d arrays
        Vector(s) of sorted trigger times (in seconds)
    bandwidth : float
        Bandwidth of the gaussian kernel
    lags : (float, float)
        Minimum and maximum lag times. Default: [-1, 1].
    npoints : int
        Number of points at which density is evaluated
    squeeze : bool
        Squeeze out singular dimensions of output array
    remove_zero_lag : bool
        Remove zero lag events
    segments : (n,2) array or Segment
        Array of time segment start and end times
    unbiased : bool
        Only include trigger events for which data is available at all lags.
        Is ignored if segments is None.

    Returns
    -------
    density : (npoints, nevents, ntriggers) array
        Peri-event densities for all event-trigger combinations
    lags : (npoints,) array
        Vector of time lags

    """
    try:
        # we prefer this one
        import compressed_kde
    except ImportError:
        try:
            # this is our fallback
            import fklab.decode as compressed_kde
        except ImportError:
            # in case neither is installed
            raise ModuleNotFoundError(name="compressed_kde")

        warnings.warn(
            "the old deprecated package of the decoding lib is used. "
            "Consider removing this one and install instead the py-compressed-kde package."
        )

    if not isinstance(events, (tuple, list)):
        events = [events]

    if not isinstance(triggers, (tuple, list)):
        triggers = [triggers]

    if lags is None:
        lags = np.array([-1, 1])
    else:
        lags = np.array(lags).ravel()

    if len(lags) != 2:
        raise ValueError("Expected (2,) sequence for lags.")

    if not segments is None:
        segments = fklab.segments.Segment(segments)
        events = [x[segments.contains(x)[0]] for x in events]
        triggers = [x[segments.contains(x)[0]] for x in triggers]

    events = [x[np.isfinite(x)] for x in events]
    triggers = [x[np.isfinite(x)] for x in triggers]

    kde_space = compressed_kde.EuclideanSpace(["lag"], bandwidth=bandwidth)
    grid_points = np.linspace(*lags, npoints)
    kde_grid = kde_space.grid([grid_points])

    yy = np.empty((npoints, len(events), len(triggers)), dtype=np.float)

    for t, trigger in enumerate(triggers):
        if not segments is None and unbiased:
            # compute segments relative to triggers
            trig_seg = np.reshape(
                np.array(segments)[:, None, :] - trigger[None, :, None],
                (len(segments) * len(trigger), 2),
            )
            ntriggers = fklab.segments.segment_count(trig_seg, grid_points)
        else:
            ntriggers = len(trigger)

        for e, event in enumerate(events):
            rel_events = event_correlation(
                event, trigger, lags=lags + [-4 * bandwidth, 4 * bandwidth]
            )[0]

            if remove_zero_lag:
                rel_events = rel_events[np.abs(rel_events) > 0]

            nrelevents = len(rel_events)

            mix = compressed_kde.Mixture(kde_space)
            mix.merge(rel_events)

            yy[:, e, t] = mix.evaluate(kde_grid) * nrelevents / ntriggers

    if squeeze:
        yy = np.squeeze(yy)

    return yy, grid_points


def spike_time_tiling_coefficient(a, b, dt=0.1, epochs=None):
    """Compute spike time tiling coefficient.

    This computes a measure of correlation between two spike
    trains. The measure is not dependent on firing rate. It was
    described in
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4205553/

    Parameters
    ----------
    a, b : 1d array
        arrays with spike times
    dt : float
        time bin
    epochs : (n,2) array
        start and stop times for the epochs within which
        the correlation needs to be computed

    Returns
    -------
    float

    """
    a = np.sort(np.asarray(a).ravel())
    b = np.sort(np.asarray(b).ravel())

    dt = float(dt)

    if epochs is None:
        epochs = [np.minimum(a[0], b[0]) - dt, np.maximum(a[-1], b[-1]) + dt]

    epochs = check_segments(epochs)
    duration = np.sum(epochs[:, -1] - epochs[:, 0])

    # define [-dt, dt] time windows around events
    sega = segment_remove_overlap(a[:, None] + [[-dt, dt]])
    segb = segment_remove_overlap(b[:, None] + [[-dt, dt]])

    # restrict to epochs
    sega = segment_intersection(sega, epochs)
    segb = segment_intersection(segb, epochs)

    # select events in epochs
    a = a[segment_contains(epochs, a)[0]]
    b = b[segment_contains(epochs, b)[0]]

    # compute fraction of time around events
    ta = np.sum(sega[:, -1] - sega[:, 0]) / duration
    tb = np.sum(segb[:, -1] - segb[:, 0]) / duration

    # compute proportion of events inside time windows
    pa = np.sum(segment_contains(segb, a)[0]) / len(a)
    pb = np.sum(segment_contains(sega, b)[0]) / len(b)

    # compute spike time tiling coefficient
    sttc = 0.5 * ((pa - tb) / (1 - pa * tb) + (pb - ta) / (1 - pb * ta))

    return sttc


def shift_events(events, shift=0, segments=None, circular=True):
    """Circularly shift event times by a fixed value.

    Parameters
    ----------
    events : 1d-array
        Array of (sorted) event times.
    shift : float
        By how much to shift the event times.
    segments : (n,2) array or Segment
        Time windows in which to perform the shift. The time windows are considered
        as a single contiguous time window for the purpose of the shift. Events
        outside the time windows are not shifted.
    circular : bool
        If True, events at the boundaries will shift to the opposite boundary. If False,
        events that are shifted outside the first/last time winodw will be removed.

    Returns
    -------
    events : 1d array
        Time-shifted events

    """
    if len(events) <= 1:
        if circular:
            return events
        else:
            return events + shift

    # more than 1 event
    if segments is None:
        # add small number to segment end, so that last event is inside the segment
        segments = fklab.segments.Segment(
            [[events[0], events[-1] + np.finfo(float).eps]]
        )
    else:
        segments = fklab.segments.Segment(segments)

    # 1. remove segment overlap
    segments = segments.removeoverlap(strict=False)
    durations = segments.duration
    duration = np.sum(durations)

    # 2. for all spikes, find corresponding segments
    isinseg, _, idx = segments.contains(events, expand=True)
    seg_idx = np.full(len(events), -1, dtype=int)
    for k, a in enumerate(idx):
        seg_idx[a] = k

    # 3. find cumulative inter-segment-intervals
    isi = np.cumsum(np.concatenate([[0], segments.start[1:] - segments.stop[:-1]]))

    # 4. adjust event times
    adjusted_events = events[isinseg] - isi[seg_idx[isinseg]]

    # 5. circularly shift spike trains
    adjusted_events += shift
    if circular:
        adjusted_events[adjusted_events >= segments.start[0] + duration] -= duration
        adjusted_events[adjusted_events < segments.start[0]] += duration

    # 6. for all adjusted spikes, find corresponding segment
    adjusted_segs = segments - isi[:, None]
    keep, _, idx = fklab.segments.segment_contains(
        adjusted_segs, adjusted_events, issorted=False, expand=True
    )
    seg_idx = np.full(len(adjusted_events), -1, dtype=int)
    for k, a in enumerate(idx):
        seg_idx[a] = k

    # 7. adjust event times
    isinseg = np.flatnonzero(isinseg)
    shifted_events = events.copy()
    shifted_events[isinseg[keep]] = adjusted_events[keep] + isi[seg_idx][keep]

    if not circular:
        shifted_events = np.delete(shifted_events, isinseg[~keep])

    return np.sort(shifted_events)
