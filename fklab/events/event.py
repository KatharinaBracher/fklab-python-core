"""Define the Event container.

.. currentmodule:: fklab.events.event

Provides a container class for a vector of event times. Event vectors
can be created and manipulated through the class. Basic event algorithms
are implemented as methods of the class.
"""
import numpy as np

import fklab.signals.core
import fklab.utilities.general
from .basic_algorithms import check_events
from .basic_algorithms import complex_spike_index
from .basic_algorithms import event_bin
from .basic_algorithms import event_bursts
from .basic_algorithms import event_count
from .basic_algorithms import event_intervals
from .basic_algorithms import event_rate
from .basic_algorithms import filter_bursts
from .basic_algorithms import filter_intervals
from .basic_algorithms import peri_event_histogram
from fklab.version._core_version._version import __version__

__all__ = ["Event"]


class AttributeMixIn:
    _attributes = {}

    def set_attribute(self, key, val):
        if len(val) != len(self):
            raise ValueError
        self._attributes[key] = val

    def get_attribute(self, key):
        return self._attributes[key]

    def list_attributes(self):
        return list(self._attributes.keys())

    def del_attribute(self, key):
        del self._attributes[key]


class Event(AttributeMixIn):
    """Define the event class.

    Attributes
    ----------
    data: array-like, 1D
    copy: bool, opt

    Examples
    --------

    **How to construct the Event object ?**

    >>> Event([1, 2, 3])
    Event(array([1, 2, 3]))

    >>> Event(np.array([1,2,3]))
    Event(array([1, 2, 3]))

    >>> Event(Event([1,2,3]))
    Event(array([1, 2, 3]))

    >>> Event([[1,2,3], [4,5,6]])
    Traceback (most recent call last):
    ...
    ValueError: Data is not arranged as a vector

    **How to manipulate the Event object ?**

    Comparaison with another event or a potential other event

    >>> Event([1,2,3]) == Event([1,2,3])
    True

    >>> Event([1,2,3]) != Event([4,5,6])
    True

    >>> Event([1,2,3]) == [1,2,3]
    True

    Comparaison with an offset

    >>> Event([1,2,3]) > 1
    array([False,  True,  True])

    Getter

    >>> Event([1,2,3])[0]
    Event(array([1]))

    Assessor

    >>> ev = Event([1,2,3])
    >>> ev[0] = 2
    >>> ev
    Event(array([2, 2, 3]))

    Iterator

    >>> for ev in Event([1,2]): print(ev)
    1
    2

    Deletion based on index :

    >>> ev = Event([1,2,3])
    >>> del ev[0]
    >>> ev
    Event(array([2, 3]))

    Deletion based on a boolean list:

    >>> ev = Event([1,2,3])
    >>> ev[ev < 2]
    Event(array([1]))

    Addition/soustraction of an offset (inplace or not):

    >>> Event([1,2,3]) + 1
    Event(array([2., 3., 4.]))

    >>> ev = Event([1,2,3])
    >>> ev += 1
    >>> ev
    Event(array([2., 3., 4.]))

    Addition of two events (inplace or not) = concatenation:

    >>> Event([1,2,3]) + Event([4,5,6])
    Event(array([1, 2, 3, 4, 5, 6]))

    >>> ev = Event([1,2,3])
    >>> ev +=  Event([4,5,6])
    >>> ev
    Event(array([1, 2, 3, 4, 5, 6]))

    Attention, it is not possible to do the substraction of two events.

    >>> Event([1,2,3]) - Event([4,5,6])
    Traceback (most recent call last):
    ...
    ValueError: Substraction of two events is not implemented.

    Other

    >>> str(Event([1,2,3]))
    'Event([1 2 3])'

    >>> len(Event([1,2,3]))
    3

    """

    def __init__(self, data=[], copy=True):

        if isinstance(data, Event):
            if copy:
                self._data = data._data.copy()
            else:
                self._data = data._data
        else:
            self._data = check_events(data, copy)

        super(Event, self).__init__()

    @staticmethod
    def isevent(x):
        """Check if x is event-like.

        As it is a static method, it can be used before creating an event object.

        >>> Event.isevent(np.array([1,2,3]))
        True

        >>> Event.isevent(Event(np.array([1,2,3])))
        True

        >>> Event.isevent([[1,2,3],[4,5,6]])
        False

        """
        if isinstance(x, Event):
            return True

        try:
            check_events(x)
        except ValueError:
            return False

        return True

    def __array__(self, *args):
        return self._data.__array__(*args)

    def asarray(self, copy=True):
        """Return numpy array representation of Event object data."""
        return self._data.copy() if copy else self._data

    def __repr__(self):
        """Return string representation of Event object."""
        return "Event(" + repr(self._data) + ")"

    def __str__(self):
        """Return string representation of Event object data."""
        return "Event(" + str(self._data) + ")"

    def __len__(self):
        """Return the number of events in the container."""
        return int(self._data.__len__())

    def issorted(self):
        """Check if events are sorted in ascending order.

        >>> Event([1,2,3]).issorted()
        True

        >>> Event([1,4,2]).issorted()
        False
        """
        return fklab.utilities.general.issorted(self._data)

    def sort(self):
        """Sort events in ascending order.

        >>> Event([1,2,3]).sort()
        Event(array([1, 2, 3]))

        >>> Event([1,4,2]).sort()
        Event(array([1, 2, 4]))

        """
        if self._data.__len__() > 1:
            idx = np.argsort(self._data)
            self._data = self._data[idx]
            # TODO: sort attributes

        return self

    def __iter__(self):
        """Iterate through events in container."""
        idx = 0
        while idx < self._data.__len__():
            yield self._data[idx]
            idx += 1

    def not_(self):
        """Test if no events are defined.

        >>> not Event([2,4])
        False
        >>> not Event([])
        True

        """
        return self._data.__len__() == 0

    def truth(self):
        """Test if one or more events are defined.

        >>> bool(Event([2,4]))
        True
        >>> bool(Event([]))
        False

        """
        return self._data.__len__() > 0

    def __eq__(self, other):
        """Test if both objects contain the same event data."""
        if not isinstance(other, Event):
            other = Event(other)
        return (self._data.shape == other._data.shape) and np.all(
            self._data == other._data
        )

    def __lt__(self, other):
        return np.asarray(self) < other

    def __le__(self, other):
        return np.asarray(self) <= other

    def __gt__(self, other):
        return np.asarray(self) > other

    def __ge__(self, other):
        return np.asarray(self) >= other

    def __getitem__(self, key):
        """Slice events."""
        return Event(self._data[key])

    def __setitem__(self, key, value):
        self._data[key] = value
        return self

    def __delitem__(self, key):
        # make sure we have a np.ndarray
        key = np.array(key)

        # if a logical vector with length equal number of segments, then find indices
        if key.dtype == np.bool and key.ndim == 1 and len(key) == self._data.__len__():
            key = np.nonzero(key)[0]

        self._data = np.delete(self._data, key, axis=0)
        return self

    def offset(self, value):
        """Add offset to events and create a new object.

        Parameters
        ----------
        value: array-like or scalar
            offset to apply to the events. In case of multiple offset, an new event array is created for each.

        Returns
        -------
        Event

        Example
        -------

        >>> Event([1,2,3]).offset(2)
        Event(array([3., 4., 5.]))

        >>> Event([1,2,3]).offset([1, 2, 3])
        Event(array([[2., 3., 4.],
               [3., 4., 5.],
               [4., 5., 6.]]))
        """
        e = Event(self)  # copy
        e.ioffset(value)  # apply offset to copy
        return e

    def ioffset(self, value):
        """Add offset to events (in place).

        Parameters
        ----------
        value: array-like or scalar
            offset to apply to the events. In case of multiple offset, an new event array is created for each.

        """
        value = np.array(value, dtype=np.float64).squeeze()

        if value.ndim == 1:
            value = value.reshape([len(value), 1])
        elif value.ndim != 0:
            raise ValueError("Invalid shape of offset value")

        self._data = self._data + value

        return self

    def __iadd__(self, value):
        """Concatenates events or adds offset (in place)."""
        if isinstance(value, Event):
            return self.iconcat(value)

        return self.ioffset(value)

    def __add__(self, value):
        """Concatenate events or adds offset."""
        if isinstance(value, Event):
            return self.concat(value)

        return self.offset(value)

    __radd__ = __add__

    def __sub__(self, value):
        """Add negative offset."""
        if isinstance(value, Event):
            raise ValueError("Substraction of two events is not implemented.")
        return self.offset(-value)

    __rsub__ = __sub__

    def __isub__(self, value):
        """Add negative offset (in place)."""
        return self.ioffset(-value)

    def concat(self, *others):
        """Concatenate events.

        Parameters
        ----------
        \*others : event vectors

        Returns
        -------
        Event

        Example
        -------

        >>> Event([1,2,3]).concat(Event([4,5,6]))
        Event(array([1, 2, 3, 4, 5, 6]))

        >>> Event([1,2,3]) + Event([4,5,6])
        Event(array([1, 2, 3, 4, 5, 6]))

        """
        e = Event(self)
        e.iconcat(*others)

        return e

    def iconcat(self, *others):
        """Concatenate events (in place).

        Parameters
        ----------
        \*others : event vectors


        Example
        -------

        >>> ev = Event([1,2,3]).iconcat(Event([4,5,6]))
        >>> ev
        Event(array([1, 2, 3, 4, 5, 6]))

        >>> Event([1,2,3]).iconcat(Event([]))
        Event(array([1., 2., 3.]))
        """
        if len(others) == 0:
            return self

        # make sure all inputs are Events
        tmp = [x if isinstance(x, Event) else Event(x) for x in others]

        # TODO: check attribute compatibility

        data = [self._data]
        data.extend([x._data for x in tmp])
        data = np.concatenate(data, axis=0)
        self._data = data

        return self

    def count(self, x):
        """Return the cumulative count of events.

        Parameters
        ----------
        x : 1d array
            times at which to evaluate cumulative event count

        Returns
        -------
        count : 1d array
            event counts

        Example
        -------

        >>> Event([1,3,4,2]).count(4)
        array([4.])

        >>> Event([1,3,4,2]).count([2, 4])
        array([2., 4.])
        """
        return event_count(self._data, x)

    def intervals(self, other=None, kind="post"):
        """Return inter-event intervals.

        Parameters
        ----------
        other : 1d array, optional
            vector of sorted event times (in seconds)
        kind : {'pre', '<', 'post', '>', 'smallest', 'largest'}
            type of interval to return. 'pre' or '<': interval to previous event,
            'post' or '>': interval to next event, 'smallest' or 'largest':
            smallest/largest of the intervals to the previous and next events.

        Returns
        -------
        intervals : 1d array
            the requested interval for each event in the input vector events.
            Intervals to events in the past have a negative sign.
        index : 1d array
            index of the event to which the interval was determined

        Example
        -------

        By default, we look for each events to the nearest post input event and give the interval to it in
        the first output and which input element has been selected in the second output.
        For example, the event 1 is closest of 2 which give us an interval of 1 and the index 0

        >>> Event([1, 2, 3 ]).intervals([2, 3])
        (array([1., 0., 0.]), array([0, 0, 1]))

        If there is no element available to compute the interval (in that case, no input element are bigger que 1, 2 and 3)
        the output will be nan for the interval, -1 for the element selected.

        >>> Event([1, 2, 3 ]).intervals([0])
        (array([nan, nan, nan]), array([-1, -1, -1]))

        With the method king "pre" or "<", we do the same thing as before except we look at the nearest input event before
        the event.

        >>> Event([1, 2, 3 ]).intervals([2], kind="pre")
        (array([nan,  0., -1.]), array([-1,  0,  0]))

        With the method king "smallest", we do the same thing as before except we look at the input event nearest
        (before and after) the event.

        >>> Event([1, 2, 3 ]).intervals([2, 3], kind="smallest")
        (array([nan,  0.,  0.]), array([-1,  0,  1]))

        With the method king "largest", we do the same thing as before except we look at the input event farest
        (before and after) the event.

        >>> Event([1, 2, 3 ]).intervals([2, 3], kind="largest")
        (array([nan,  0.,  0.]), array([-1,  0,  1]))

        """
        return event_intervals(self._data, other=other, kind=kind)

    def bin(self, bins, kind="count"):
        """Count number of events in bins.

        Parameters
        ----------
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

        Example
        -------
        **Count method:** (default method)

        >>> Event(np.array([2.5, 3, 4, 1, 1.5, 2, 5, 6, 7, 8])).bin([[1, 3], [3, 7]])
        array([[4],
               [4]], dtype=uint64)

        **Binary method:**

        >>> Event(np.array([2.5, 3, 4, 1, 1.5, 2, 5, 6, 7, 8])).bin([[1, 3], [3, 7]], kind="rate")
        array([[2.],
               [1.]])

        **Rate method:**

        >>> Event(np.array([2.5, 3, 4, 1, 1.5, 2, 5, 6, 7, 8])).bin([[1, 3], [3, 7]], kind="rate")
        array([[2.],
               [1.]])

        Attention, each bin pair needs to be sorted because it is transfomred as a segment = [start, stop times[.

        >>> Event(np.array([2.5, 3])).bin([3, 1])
        Traceback (most recent call last):
        ...
        ValueError: Segment durations cannot be negative

        However, the list of bin pairs itself does not need to be sorted.

        >>> Event(np.array([2.5, 3, 4, 1, 1.5, 2, 5, 6, 7, 8])).bin([[3, 7], [1, 3]], kind="rate")
        array([[1.],
               [2.]])
        """
        return event_bin(self._data, bins, kind=kind)

    def meanrate(self, segments=None, separate=False):
        """Return mean rate of events.

        Parameters
        ----------
        segments : (n,2) sorted array or Segment, optional
            array of time segment start and end times

        separate: bool, opt - compute for separate input segments

        Returns
        -------
        rate : array
            Mean firing rate

        Example
        -------

        >>> Event(np.array([1, 1.5, 2, 2.5, 3, 4])).meanrate([[1, 3]])
        array([2.])

        With multiple segments, the result by default give the mean rate of the combined input segments.

        >>> Event(np.array([1, 1.5, 2, 2.5, 3, 4])).meanrate([[1, 3], [3,4]])
        array([1.66666667])

        To obtain, the event rate for each input segments, you need to set the option separate to True.

        >>> Event(np.array([1, 1.5, 2, 2.5, 3, 4])).meanrate([[1, 3], [3,4]], separate=True)
        array([[2., 1.]])

        """
        return event_rate(self._data, segments=segments, separate=separate)

    def peri_event_histogram(
        self,
        reference=None,
        lags=None,
        segments=None,
        normalization=None,
        unbiased=False,
        remove_zero_lag=False,
    ):
        """Compute peri-event time histogram.

        Parameters
        ----------
        reference : 1d array or sequence of 1d arrays, optional
            vector(s) of sorted reference event times (in seconds).
            If not provided, then events are used as a reference.
        lags : 1d array, optional
            vector of sorted lag times that specify the histogram time bins
        segments : (n,2) array or Segment, optional
            array of time segment start and end times
        normalization : {'none', 'coef', 'rate', 'conditional mean intensity','product density', 'cross covariance','cumulant density', 'zscore'}, optional
            type of normalization
        unbiased : bool, optional
            only include reference events for which data is available at all lags
        remove_zero_lag : bool, optional
            remove zero lag event counts

        Returns
        -------
        3d array
            peri-event histogram of shape (lags, events, references)

        """
        return peri_event_histogram(
            self._data,
            reference=reference,
            lags=lags,
            segments=segments,
            normalization=normalization,
            unbiased=unbiased,
            remove_zero_lag=remove_zero_lag,
        )

    def peri_event_density(
        self,
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

        return peri_event_density(
            self._data,
            triggers,
            lags,
            npoints,
            bandwidth,
            squeeze,
            remove_zero_lag,
            segments,
            unbiased,
        )

    def detectbursts(
        self, intervals=None, nevents=None, amplitude=None, attribute=False
    ):
        """Detect bursts of events.

        Parameters
        ----------
        intervals : 2-element sequence, optional
            minimum and maximum inter-event time intervals to consider two
            events as part of a burst
        nevents : 2-element sequence, optional
            minimum and maximum number of events in a burst
        amplitude : 1d array, optional
            vector of event amplitudes
        attribute: bool
            this parameter is not used at this time

        Returns
        -------
        1d array
            vector with burst indicators: 0=non-burst event, 1=first event
            in burst, 2=event in burst, 3=last event in burst


        Examples
        --------

        >>> Event([2,2.5,3,4,5,6,6.5,7]).detectbursts(intervals=[0.9, 1.1])
        array([0., 0., 1., 2., 2., 3., 0., 0.])

        >>> Event([2,2.5,3,4,5,6,6.5,7]).detectbursts(intervals=[0.9, 1.1], nevents=[3,7])
        array([0., 0., 1., 2., 2., 3., 0., 0.])

        Adding the amplitude of each event in input allow to detect complex spike burst.

        """
        return event_bursts(
            self._data, intervals=intervals, nevents=nevents, marks=amplitude
        )

    def filterbursts(self, bursts=None, method="reduce", **kwargs):
        """Filter events in place based on participation in bursts.

        Parameters
        ----------
        bursts : 1d array, optional
            burst indicator vector as returned by event_bursts function.
            If not provided, it will be computed internally (parameters to
            the event_bursts function can be provided as extra keyword arguments)
        method : {'none', 'reduce', 'remove', 'isolate', 'isolatereduce'}
            filter method to be applied.

        \*\*kwargs: dict
            Arguments for the event_burst method


        See Also
        --------
        fklab.events.basic_algorithms.event_burst

        Example
        -------

        **'reduce': only keep non-burst events and first event in bursts**

        >>> Event([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]).filterbursts(method="reduce",intervals=[0.9, 1.1])
        Event(array([1. , 2.5, 3. , 6.5, 7. ]))

        **'remove': remove all burst events**

        >>> Event([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]).filterbursts(method="remove",intervals=[0.9, 1.1])
        Event(array([2.5, 6.5, 7. ]))

        **'isolate': remove all non-burst events**

        >>> Event([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]).filterbursts(method="isolate",intervals=[0.9, 1.1])
        Event(array([1., 2., 3., 4., 5., 6.]))

        **'isolatereduce': only keep first event in bursts**

        >>> Event([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]).filterbursts(method="isolatereduce",intervals=[0.9, 1.1])
        Event(array([1., 3.]))

        **'none' or any other keyword: Do nothing - Keep all events**

        >>> Event([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]).filterbursts(method="none",intervals=[0.9, 1.1])
        Event(array([1. , 2. , 2.5, 3. , 4. , 5. , 6. , 6.5, 7. ]))

        """
        events, idx = filter_bursts(self._data, bursts=bursts, method=method, **kwargs)
        self._data = events
        return self
        # TODO: filter attributes, allow bursts='attribute'

    def filterintervals(self, mininterval=0.003):
        """Filter out events in place based on interval to previous event.

        Parameters
        ----------
        mininterval : scalar, optional

        Returns
        -------
        Event
            itself (inplace manipulation)

        Example
        -------

        >>> Event([1, 1.1, 2, 3]).filterintervals(mininterval=0.5)
        Event(array([1., 2., 3.]))

        >>> Event([1, 2, 3, 1.1, -1]).filterintervals(mininterval=0.5)
        Traceback (most recent call last):
        ...
        ValueError: The events should be sorted before being filtered to avoid unexpected behavior.

        """
        if not self.issorted():
            raise ValueError(
                "The events should be sorted before being filtered "
                "to avoid unexpected behavior."
            )

        events, idx = filter_intervals(self._data, mininterval=mininterval)
        self._data = events
        return self
        # TODO: filter attributes

    def density(self, x=None, kernel="gaussian", bandwidth=0.1, rtol=0.05, **kwargs):
        """Estimate the Kernel density.

        Parameters
        ----------
        x : ndarray
            Time points at which to evaluate density. If x is None, then a KernelDensity object is returned.
        kernel : str
        bandwidth : scalar
            Kernel bandwidth
        rtol, \*\*kwargs : extra arguments for sklearn.neighbor.kde.KernelDensity.

        Returns
        -------
        ndarray or KernelDensity object


        See also
        --------
        sklearn.neighbor.kde.KernelDensity

        """
        from sklearn.neighbors.kde import KernelDensity

        # create density function
        # TODO test speed of KernelDensity - roll our own?
        kde = KernelDensity(
            kernel=kernel, bandwidth=bandwidth, rtol=rtol, **kwargs
        ).fit(self._data[:, None])
        if x is not None:
            # evaluate density function
            x = np.array(x, copy=False)
            kde = np.exp(kde.score_samples(x[:, None]))
        return kde

    def complex_spike_index(self, amplitude=None, intervals=None):
        """Compute complex spike index.

        Parameters
        ----------
        amplitude : 1d array
            vector of spike amplitudes
        intervals : 2-element sequence
            minimum and maximum inter-spike time intervals to consider two
            spikes as part of a burst

        Returns
        -------
        scalar
            coefficient criterion

        References
        ----------
        Add what is a complex spike in burst
        (burst spikes with descending amplitudes)

        """
        return float(
            complex_spike_index(self._data, spike_amp=amplitude, intervals=intervals)
        )

    def average(
        self, time, data, lags=None, fs=None, interpolation="linear", function=None
    ):
        """Average of the signal around a certain event time.

        Parameters
        ----------
        time : 1d array
            vector of sample times for data array
        data : ndarray
            array of data samples. First dimension should be time.
        lags : 2-element sequence, optional
            minimum and maximum lags over which to compute average. default = [-1, 1]
        fs : float, optional
            sampling frequency of average. If not provided, will be calculated from time vector t
        interpolation : string or integer, optional
            kind of interpolation.
        function : callable, optional
            function to apply to data samples (e.g. to compute something else
            than the average), needs to accept the axis option. Default is np.nanmean

        Returns
        -------
        ndarray
            event triggered average of data
        ndarray
            vector of lags


        See also
        --------
        scipy.interpolate.interp1d


        Example
        -------

        >>> Event([1]).average([0,1,2,3,4], [0,1, 1, 0, 0])
        (array([0., 1.]), array([-1.,  0.]))

        """
        return fklab.signals.core.event_triggered_average(
            self._data,
            time,
            data,
            lags=lags,
            fs=fs,
            interpolation=interpolation,
            function=function,
        )


# annotations - dict of container-level metadata - initiate with read-only keys
# attributes - dict of item-level metadata (len(attr)==nitems)

# obj.annotations[key] -> returns annotation

# obj.attributes[key] -> returns attribute
# obj.attributes[key][idx] -> should be valid and return subset of attributes
# len(obj.attributes[key]) -> should be equal to len(obj)

# obj[idx] -> returns subset of items and their attributes

# split_by_attribute(attribute) -> returns list of objects with subset of original items, split by attribute value ( equality, ranges, custom function ) in case of equality, optionally turn attribute into annotation
# select_by_attribute(s) -> returns subset of items where fcn( attribute value, *args ) is True

# at object collection level
# select_by_annotation( ... )

# setting/concatenation of containers should also check attributes and merge annotations

# class ValidationDict(dict):

# def __init__(self, reference, *args, **kwargs):
# self._reference = reference
# try:
# len(self._reference)
# except:
# raise TypeError

# self.update(*args, **kwargs)

##def __setitem__(self, key, value):
##    if key in self._readonlykeys:
##        raise KeyError("%(key)s is a read-only key" % {'key':str(key)})
##    super(ValidationDict, self).__setitem__(key, value)

# def __setitem__(self, key, value):
# if len(value)!=len(self._reference):
# raise ValueError("Incorrect len of value")
# super(ValidationDict, self).__setitem__(key, value)

# def update(self, *args, **kwargs):
# if args:
# if len(args) > 1:
# raise TypeError("update expected at most 1 arguments, "
# "got %d" % len(args))
# other = dict(args[0])
# for key in other:
# self[key] = other[key]
# for key in kwargs:
# self[key] = kwargs[key]

# def setdefault(self, key, value=None):
# if key not in self:
# self[key] = value
# return self[key]

# class ContainerBase(object):

# def __init__(self):
# self.attributes = ValidationDict( self )
# self.annotations = {}

# def __len__(self):
# raise NotImplementedError
