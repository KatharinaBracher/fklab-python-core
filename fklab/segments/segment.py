"""
=============================================
Segment class (:mod:`fklab.segments.segment`)
=============================================

.. currentmodule:: fklab.segments.segment

Provides a container class for a list of segments. Each segment is defined by
a start and end point (usually in units of time). Basic operations on lists
of segments are implemented as methods of the class.

"""
import numpy as np
import scipy.interpolate

from .basic_algorithms import check_segments
from .basic_algorithms import segment_applyfcn
from .basic_algorithms import segment_asindex
from .basic_algorithms import segment_concatenate
from .basic_algorithms import segment_contains
from .basic_algorithms import segment_count
from .basic_algorithms import segment_difference
from .basic_algorithms import segment_exclusive
from .basic_algorithms import segment_has_overlap
from .basic_algorithms import segment_intersection
from .basic_algorithms import segment_invert
from .basic_algorithms import segment_join
from .basic_algorithms import segment_overlap
from .basic_algorithms import segment_remove_overlap
from .basic_algorithms import segment_scale
from .basic_algorithms import segment_sort
from .basic_algorithms import segment_span
from .basic_algorithms import segment_split
from .basic_algorithms import segment_uniform_random
from .basic_algorithms import segment_union
from fklab.codetools import deprecated
from fklab.utilities.general import issorted
from fklab.utilities.general import partition_vector
from fklab.version._core_version._version import __version__


__all__ = ["SegmentError", "Segment"]


class SegmentError(Exception):
    """Exception raised if array does not represent segments."""

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)


class Segment(object):
    """Segment container class.

    Parameters
    ----------
    data : Segment, array_like
        array to be translated as segment with start/stop value

    Example
    -------

    **How to construct the Segment object ?**

    Create a segment from a list :

    >>> Segment([1, 2])
    Segment(array([[1, 2]]))

    >>> Segment([[1, 2], [3, 4], [5, 6]])
    Segment(array([[1, 2],
           [3, 4],
           [5, 6]]))

    Create a segment from a numpy array:

    >>> Segment(np.ones((4, 2)))
    Segment(array([[1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.]]))


    **How to manipulate the Segment object ?**

    Comparison with another segment or a potential other segment

    >>> Segment([[1,2], [2,3]]) ==  Segment([[1,2], [2,3]])
    True

    >>> Segment([1,2,3]) !=  Segment([[1,2], [4,6]])
    True

    >>> Segment([1,2]) == [1,2]
    True

    Boolean logical manipulation

    >>> Segment([1, 4]) | Segment([3, 7])
    Segment(array([[1., 7.]]))

    >>> Segment([1, 4]) & Segment([3, 7])
    Segment(array([[3., 4.]]))

    >>> Segment([1,7]) ^ Segment([5, 9])
    Segment(array([[1., 5.],
           [7., 9.]]))

    >>> ~Segment([2,4])
    Segment(array([[-inf,   2.],
           [  4.,  inf]]))

    Get/Set

    >>> Segment([[1,2], [3,6]])[0]
    Segment(array([[1, 2]]))

    >>> seg = Segment([[1,2], [3,6]])
    >>> seg[0] = [2, 4]
    >>> seg
    Segment(array([[2, 4],
           [3, 6]]))

    Iterator

    >>> for seg in Segment([[1,2], [2,4]]): print(seg)
    (1, 2)
    (2, 4)

    Select by index

    >>> seg = Segment([[1,2], [3,6]])
    >>> del seg[1]
    >>> seg
    Segment(array([[1, 2]]))

    Select based on a boolean list:

    >>> seg = Segment([[1,2], [3,6]])
    >>> seg[seg.start < 2]
    Segment(array([[1, 2]]))

    Concatenation of two event series (in place or not):

    >>> Segment([[1,2],[2,3]]) + Segment([3, 4])
    Segment(array([[1, 2],
           [2, 3],
           [3, 4]]))

    Addition/substraction of an offset (inplace or not):

    >>> Segment([[1,2],[2,3]]) + 3
    Segment(array([[4., 5.],
           [5., 6.]]))

    >>> Segment([[1,2],[2,3]]) - 3
    Segment(array([[-2., -1.],
           [-1.,  0.]]))

    Scaling

    >>> Segment([[1,2],[2,3]])/2
    Segment(array([[1.25, 1.75],
           [2.25, 2.75]]))

    Others

    >>> str(Segment([1,2]))
    'Segment([[1 2]])'

    >>> len(Segment([[1,2],[2,4]]))
    2
    """

    def __init__(self, data=[], copy=True):
        if isinstance(data, Segment):
            if copy:
                self._data = data._data.copy()
            else:
                self._data = data._data
        else:
            self._data = check_segments(data, copy)

    @staticmethod
    def issegment(x):
        """Test is x is valid segment array.

        >>> Segment.issegment("a")
        False

        >>> Segment.issegment([[0, 10], [20, 30], [50, 100]])
        True
        """
        if isinstance(x, Segment):
            return True

        try:
            check_segments(x)
        except ValueError:
            return False
        return True

    @classmethod
    def fromlogical(cls, y, x=None, interpolate=False):
        """Construct Segment from logical vector.

        Parameters
        ----------
        y : 1d logical array
            Any sequenceof True values that is flanked by False values is
            converted into a segment.
        x : 1d array like, optional
            The segment indices from y will be used to index into x.
        interpolate: bool, optional
            if true, segments of duration 0 are extent to have a minimal duration

        Returns
        -------
        Segment

        Examples
        --------

        Very practical to obtain segment of signal above a certain threshold for example:

        >>> Segment.fromlogical(np.array([10, 5, 12, 10, 3]) > 8)
        Segment(array([[0, 0],
               [2, 3]]))

        >>> Segment.fromlogical(np.array([10, 5, 12, 10, 3]) > 15)
        Segment(array([], shape=(0, 2), dtype=int64))

        Or to obtain the time segment corresponding to the signal above the threshold

        >>> Segment.fromlogical(np.array([10, 5, 12, 10, 3]) > 8, x=np.array([2, 4, 6, 8, 10]))
        Segment(array([[2, 2],
               [6, 8]]))

        >>> Segment.fromlogical(np.array([10, 5, 12, 10, 3]) < 8, x=np.array([2, 4, 6, 8, 10]), interpolate = True)
        Segment(array([[ 3.,  5.],
               [ 9., 10.]]))
        """
        y = np.asarray(y == True, dtype=np.int8).ravel()

        if len(y) == 0 or np.all(y == 0):
            return cls(np.zeros((0, 2), dtype=np.int64))

        offset = 0
        if interpolate:
            offset = 0.5

        d = np.diff(np.concatenate(([0], y, [0])))
        segstart = np.nonzero(d[0:-1] == 1)[0] - offset
        segend = np.nonzero(d[1:] == -1)[0] + offset

        seg = np.vstack((segstart, segend)).T

        if x is not None:
            if interpolate:
                seg = scipy.interpolate.interp1d(
                    np.arange(len(y)),
                    x,
                    kind="linear",
                    bounds_error=False,
                    fill_value=(x[0], x[-1]),
                )(seg)
            else:
                seg = x[seg]

        return cls(seg)

    @classmethod
    def fromindices(cls, y, x=None):
        """Construct segments from vector of indices.

        Parameters
        ----------
        y : 1d array like
            Vector of indices. Segments are created from all neighboring
            pairs of values in y (as long as the difference is positive).
        x : 1d array like, optional
            The segment indices from y will be used to index into x.

        Returns
        -------
        Segment

        Examples
        --------

        >>> Segment.fromindices(np.array([3, 6, 7, 9]))
        Segment(array([[3, 3],
               [6, 7],
               [9, 9]]))

        >>> Segment.fromindices([])
        Segment(array([], shape=(0, 2), dtype=float64))


        >>> Segment.fromindices(np.array([3, 6, 7, 9]), x=np.arange(start=0, stop=20, step=2))
        Segment(array([[ 6,  6],
               [12, 14],
               [18, 18]]))
        """
        if len(y) == 0:
            return cls([])

        d = np.nonzero(np.diff(y) > 1)[0]
        segstart = y[np.concatenate(([0], d + 1))]
        segend = y[np.concatenate((d, [len(y) - 1]))]

        seg = np.vstack((segstart, segend)).T

        if x is not None:
            seg = x[seg]

        return cls(seg)

    @classmethod
    def fromevents(cls, on, off, greedyStart=False, greedyStop=False):
        """Construct segments from sequences of start and stop values.

        Parameters
        ----------
        on : 1d array like
            segment start values.
        off : 1d array like
            segment stop values.
        greedyStart : bool
            If multiple start values precede a stop value, then the first
            start value is used.
        greedyStop : bool
            If multiple stop values follow a start value, then the last
            stop value is used.

        Returns
        -------
        Segment


        Examples
        --------
        >>> Segment.fromevents(10, 20)
        Segment(array([[10., 20.]]))

        >>> Segment.fromevents([10, 15], [20, 30])
        Segment(array([[15., 20.]]))

        >>> Segment.fromevents([10, 15], [20, 30], greedyStart=True)
        Segment(array([[10., 20.]]))

        >>> Segment.fromevents([10, 15], [20, 30], greedyStop=True)
        Segment(array([[15., 30.]]))
        """
        on = np.array(on, dtype=np.float64).ravel()
        off = np.array(off, dtype=np.float64).ravel()

        events = np.concatenate((on, off))
        eventid = np.concatenate((np.ones(len(on)), -np.ones(len(off))))

        isort = np.argsort(
            events, kind="mergesort"
        )  # mergesort keeps items with same key in same relative order
        events = events[isort]
        eventid = eventid[isort]

        diff_eventid = np.diff(eventid)

        # if greedyStart = True, remove all on-events in blocks (except first one)
        if greedyStart:
            invalid = (
                np.nonzero(np.logical_and(diff_eventid == 0, eventid[1:] == 1))[0] + 1
            )
        else:
            invalid = np.nonzero(np.logical_and(diff_eventid == 0, eventid[0:-1] == 1))[
                0
            ]

        # if greedyStop = True, remove all off-events in blocks (except last one)
        if greedyStop:
            invalid = np.concatenate(
                (
                    invalid,
                    np.nonzero(np.logical_and(diff_eventid == 0, eventid[0:-1] == -1))[
                        0
                    ],
                )
            )
        else:
            invalid = np.concatenate(
                (
                    invalid,
                    np.nonzero(np.logical_and(diff_eventid == 0, eventid[1:] == -1))[0]
                    + 1,
                )
            )

        events = np.delete(events, invalid)
        eventid = np.delete(eventid, invalid)

        s = np.nonzero(np.diff(eventid) == -2)[0]
        s = np.vstack((events[s], events[s + 1])).T

        return cls(s)

    @classmethod
    def fromduration(cls, anchor, duration, reference=0.5):
        """Construct segments from anchor points and durations.

        Parameters
        ----------
        anchor : scalar or 1d array like
            Anchoring points for the new segments. If reference is not given, then the anchor determines the segment center.
        duration : scalar or 1d array like
            Durations of the new segments
        reference : scalar or 1d array like, optional
            Relative reference point of the anchor in the segment. If reference is 0., the anchor defines the segment start,
            if reference is 1., the anchor defines the segment stop.

        Returns
        -------
        Segment


        Examples
        --------
        The anchor is at the middle of the segment (default mode)

        >>> Segment.fromduration(10, 20)
        Segment(array([[ 0., 20.]]))

        >>> Segment.fromduration(10, 20, reference = 0)
        Segment(array([[10., 30.]]))

        The anchor is at the end of the segment because the reference=1

        >>> Segment.fromduration(10, 20, reference = 1)
        Segment(array([[-10.,  10.]]))

        """
        # anchor + duration*[-reference (1-reference)]
        anchor = np.array(anchor, dtype=np.float64).ravel()
        duration = np.array(duration, dtype=np.float64).ravel()
        reference = np.array(reference, dtype=np.float64).ravel()

        start = anchor - reference * duration
        stop = anchor + (1 - reference) * duration

        return cls(np.vstack((start, stop)).T)

    def __array__(self, *args):
        return self._data.__array__(*args)

    @deprecated("Please use np.asarray(obj) instead.")
    def asarray(self, copy=True):
        """Return numpy array representation of Segment object data.

        Parameters
        ----------
        copy : bool, optional
        """

        return self._data.copy() if copy else self._data

    def __repr__(self):
        """Return string representation of Segment object."""
        return "Segment(" + repr(self._data) + ")"

    def __str__(self):
        """Return string representation of Segment object data."""
        return "Segment(" + str(self._data) + ")"

    def span(self):
        """Span of segments.

        Returns
        -------
        segment that spans input segments

        Examples
        --------

        >>> Segment([[1, 2], [4, 5]]).span()
        Segment(array([[1, 5]]))

        >>> Segment([]).span()
        Segment(array([], shape=(0, 2), dtype=float64))
        """
        return Segment(segment_span(self._data))

    @property
    def start(self):
        """Get/Set vector of segment start values.

        Examples
        --------

        >>> Segment([[1, 2], [4, 5]]).start
        array([1, 4])

        >>> test = Segment([[1, 2], [4, 5]])
        >>> test.start = np.array([0, 2])
        >>> test
        Segment(array([[0, 2],
               [2, 5]]))

        .. note:: sif start > stop for any segment then a SegmentError is raised

        >>> test = Segment([[1, 2], [4, 5]])
        >>> test.start = np.array([3, 2])
        Traceback (most recent call last):
        ...
        fklab.segments.segment.SegmentError: Segment start times should be <= stop times
        """
        return self._data[:, 0].copy()

    @start.setter
    def start(self, value):
        # Should we re-order after changing start points?
        if np.any(self._data[:, 1] < value):
            raise SegmentError("Segment start times should be <= stop times")
        self._data[:, 0] = value

    @property
    def stop(self):
        """Get/Set vector of segment stop values.

        Examples
        --------

        >>> Segment([[1, 2], [4, 5]]).stop
        array([2, 5])

        >>> test = Segment([[1, 2], [4, 5]])
        >>> test.stop = np.array([3, 6])
        >>> test
        Segment(array([[1, 3],
               [4, 6]]))

        .. note:: a SegmentError is raised if stop > start for any segment

        >>> test = Segment([[1, 2], [4, 5]])
        >>> test.stop = np.array([0, 2])
        Traceback (most recent call last):
        ...
        fklab.segments.segment.SegmentError: Segment stop times should be >= start times
        """
        return self._data[:, 1].copy()

    @stop.setter
    def stop(self, value):
        if np.any(self._data[:, 0] > value):
            raise SegmentError("Segment stop times should be >= start times")

        self._data[:, 1] = value

    @property
    def duration(self):
        """Get/set a vector of segment durations.

        Examples
        --------

        >>> Segment([1, 3, 6]).duration
        array([2, 3])

        The setter recomputes the new segment to have the requested duration while
        keeping the same center as previously.

        >>> seg = Segment([[1, 3], [4, 5]])
        >>> seg.duration = np.array([2, 6])
        >>> seg
        Segment(array([[1, 3],
               [1, 7]]))

        """
        return np.diff(self._data, axis=1).ravel()

    @duration.setter
    def duration(self, value):
        value = np.array(value, dtype=np.float64).ravel()
        ctr = np.mean(self._data, axis=1)
        self._data[:, 0] = ctr - 0.5 * value
        self._data[:, 1] = ctr + 0.5 * value

    @property
    def center(self):
        """Get/Set segment centers.

        Examples
        --------

        >>> Segment([1, 3, 6]).center
        array([2. , 4.5])

        The setter recompute the new segment to have the requested center while
        keeping the same duration as previously.

        >>> seg = Segment([[1, 3], [4, 5]])
        >>> seg.center = np.array([2, 6])
        >>> seg
        Segment(array([[1, 3],
               [5, 6]]))

        """
        return np.mean(self._data, axis=1)

    @center.setter
    def center(self, value):
        value = np.array(value, dtype=np.float64).ravel()
        dur = np.diff(self._data, axis=1).squeeze()
        self._data[:, 0] = value - 0.5 * dur
        self._data[:, 1] = value + 0.5 * dur

    def __len__(self):
        """Return the number of segments in the container."""
        return int(self._data.shape[0])

    def issorted(self):
        """Check if segment starts are sorted in ascending order.

        Examples
        --------

        >>> Segment([[2,4]]).issorted()
        True

        >>> Segment([[0,4], [1,5], [3, 6]]).issorted()
        True

        >>> Segment([[2,4], [1,5], [3, 6]]).issorted()
        False

        See Also
        --------
        fklab.utilities.general.issorted

        """
        return issorted(self._data[:, 0])

    def isort(self):
        """Sort segments (in place) in ascending order according to start value.

        Returns
        -------
        self
             itself sorted

        Examples
        --------

        >>> Segment([[2,4], [1,5]]).isort()
        Segment(array([[1, 5],
               [2, 4]]))
        """

        self._data = segment_sort(self._data)
        return self

    def sort(self):
        """Sort segments in ascending order according to start value.

        Returns
        -------
        Segment
             new segment

        Examples
        --------

        >>> Segment([[2,4], [1,5]]).sort()
        Segment(array([[1, 5],
               [2, 4]]))
        """
        s = Segment(self)
        s.isort()
        return s

    def argsort(self):
        """Indices of the sorted segment by the start value.

        Returns
        -------
        ndarray
            Indices that will sort the segment array.

        Examples
        --------

        >>> Segment([[2,4], [1,5]]).argsort()
        array([1, 0])
        """
        return np.argsort(self._data[:, 0])

    @property
    def intervals(self):
        """Duration of intervals between segments.

        Examples
        --------

        >>> Segment([[2,4], [1,5]]).intervals
        array([-3])
        """
        return self._data[1:, 0] - self._data[:-1, 1]

    def hasoverlap(self):
        """Check if any segments are overlapping.

        Returns
        -------
        bool
            True if any of the segments overlap.

        Examples
        --------

        >>> Segment([[2,4], [1,5]]).hasoverlap()
        True

        >>> Segment([[3,4], [1,2]]).hasoverlap()
        False

        """
        return segment_has_overlap(self._data)

    def removeoverlap(self, strict=True):
        """Remove overlap between segments through merging.

        This method will sort segments as a side effect.

        Parameters
        ----------
        strict : bool
            Only merge two segments if the end time of the first is stricly
            larger than (and not equal to) the start time of the second segment.

        Returns
        -------
        self
            return itself for chained methods

        Examples
        --------

        >>> Segment([[2,4], [1,5]]).removeoverlap()
        Segment(array([[1, 5]]))

        >>> Segment([[2,4], [1,2]]).removeoverlap(strict=False)
        Segment(array([[1, 2],
               [2, 4]]))

        """
        self._data = segment_remove_overlap(self._data, strict=strict)
        return self

    def __iter__(self):
        """Iterate through segments in container."""
        idx = 0
        while idx < self._data.shape[0]:
            yield self._data[idx, 0], self._data[idx, 1]
            idx += 1

    def not_(self):
        """Test if no segments are defined.

        Examples
        --------

        >>> not Segment([[2,4], [1,5]])
        False
        >>> not Segment([])
        True
        """
        return self._data.shape[0] == 0

    def truth(self):
        """Test if one or more segments are defined.

        Examples
        --------

        >>> bool(Segment([[2,4], [1,5]]))
        True
        >>> bool(Segment([]))
        False
        """
        return self._data.shape[0] > 0

    def exclusive(self, *others):
        """Exclude other segments.

        Extracts parts of segments that do not overlap with any other
        segment. Will remove overlaps as a side effect.

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        Example
        -------

        >>> Segment([1, 6]).exclusive([2,4])
        Segment(array([[1., 2.],
               [4., 6.]]))
        """
        s = Segment(self)
        s.iexclusive(*others)
        return s

    def iexclusive(self, *others):
        """Exclude other segments (in place).

        Extracts parts of segments that do not overlap with any other
        segment. Will remove overlaps as a side effect.

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_exclusive(self._data, *others)
        return self

    def invert(self):
        """Invert segments.

        Constructs segments from the inter-segment intervals.
        This method will remove overlap as a side effect.

        Returns
        -------
        Segment

        Examples
        --------

        >>> Segment([[1,2], [4,5]]).invert()
        Segment(array([[-inf,   1.],
               [  2.,   4.],
               [  5.,  inf]]))
        """
        s = Segment(self)
        s.iinvert()
        return s

    def iinvert(self):
        """Invert segments (in place).

        Constructs segments from the inter-segment intervals.
        This method will remove overlap as a side effect.
        """
        self._data = segment_invert(self._data)
        return self

    __invert__ = invert

    def union(self, *others):
        """Combine segments (logical OR).

        This method will remove overlaps as a side effect.

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        >>> Segment([1,5]).union([2,7])
        Segment(array([[1., 7.]]))

        """
        s = Segment(self)
        s.iunion(*others)
        return s

    def iunion(self, *others):
        """Combine segments (logical OR) (in place).

        This method Will remove overlaps as a sife effect.

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_union(self._data, *others)
        return self

    __or__ = union
    __ror__ = __or__
    __ior__ = iunion

    def difference(self, *others):
        """Return non-overlapping parts of segments (logical XOR).

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        >>> Segment([1,7]).difference([5, 9])
        Segment(array([[1., 5.],
               [7., 9.]]))
        """
        s = Segment(self)
        s.idifference(*others)
        return s

    def idifference(self, *others):
        """Return non-overlapping parts of segments (logical XOR) (in place).

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_difference(self._data, *others)
        return self

    __xor__ = difference
    __rxor__ = __xor__
    __ixor__ = idifference

    def intersection(self, *others):
        """Return intersection (logical AND) of segments.

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        >>> Segment([1, 7]).intersection([2, 9])
        Segment(array([[2., 7.]]))

        """
        s = Segment(self)
        s.iintersection(*others)
        return s

    def iintersection(self, *others):
        """Return intersection (logical AND) of segments (in place).

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_intersection(self._data, *others)
        return self

    __and__ = intersection
    __rand__ = __and__
    __iand__ = iintersection

    def __eq__(self, other):
        """Test if both objects contain the same segment data.

        Parameters
        ----------
        other : segment array

        Returns
        -------
        bool

        """
        if not isinstance(other, Segment):
            other = Segment(other)
        return (self._data.shape == other._data.shape) and np.all(
            self._data == other._data
        )

    def __getitem__(self, key):
        """Slice segments.

        Parameters
        ----------
        key : slice or indices

        Returns
        -------
        Segment

        """
        return Segment(self._data[key, :])  # does not return a view!

    def __setitem__(self, key, value):
        """Set segment values.

        Parameters
        ----------
        key : slice or indices
        value : scalar or ndarray

        """
        self._data[key, :] = value
        return self

    def __delitem__(self, key):
        """Delete segments (in place).

        Parameters
        ----------
        key : array like
            Index vector or boolean vector that indicates which segments
            to delete.

        """
        # make sure we have a np.ndarray
        key = np.array(key)

        # if a logical vector with length equal number of segments, then find indices
        if key.dtype == bool and key.ndim == 1 and len(key) == self._data.shape[0]:
            key = np.nonzero(key)[0]

        self._data = np.delete(self._data, key, axis=0)
        return self

    def offset(self, value):
        """Add offset to segments.

        Parameters
        ----------
        value : scalar or 1d array

        Returns
        -------
        Segment

        Examples
        --------

        >>> Segment([[1,2],[2,3]]).offset(3)
        Segment(array([[4., 5.],
               [5., 6.]]))

        In case of offset array, first scalar is added to the first segment ...etc.

        >>> Segment([[1,2],[2,3]]).offset([3, 2])
        Segment(array([[4., 5.],
               [4., 5.]]))
        """
        s = Segment(self)
        s.ioffset(value)

        return s

    def ioffset(self, value):
        """Add offset to segments (in place).

        Parameters
        ----------
        value : scalar or 1d array

        """
        value = np.array(value, dtype=np.float64).squeeze()

        if value.ndim == 1:
            value = value.reshape([len(value), 1])
        elif value.ndim != 0:
            raise ValueError("Invalid shape of offset value")

        self._data = self._data + value
        return self

    def scale(self, *args, **kwargs):
        """Scale segment durations.

        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor
        reference: scalar or 1d array
            Relative reference point in segment used for scaling. A value of
            0.5 means symmetrical scaling around the segment center. A value
            of 0. means that the segment duration will be scaled without
            altering the start time.

        Returns
        -------
        Segment

        Examples
        --------

        >>> Segment([[1,2],[2,3]]).scale(3)
        Segment(array([[0., 3.],
               [1., 4.]]))

        >>> Segment([[1,2],[2,3]]).scale(3, reference=0)
        Segment(array([[1., 4.],
               [2., 5.]]))
        """
        s = Segment(self)
        s.iscale(*args, **kwargs)
        return s

    def iscale(self, value, reference=0.5):
        """Scale segment durations (in place).

        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor
        reference: scalar or 1d array
            Relative reference point in segment used for scaling. A value of
            0.5 means symmetrical scaling around the segment center. A value
            of 0. means that the segment duration will be scaled without
            altering the start time.

        """
        self._data = segment_scale(self._data, value, reference=reference)
        return self

    def concat(self, *others):
        """Concatenate segments.

        Parameters
        ----------
        *others : segment arrays

        Returns
        -------
        Segment

        Examples
        --------

        >>> Segment([[1,2],[2,3]]).concat(Segment([3,4]))
        Segment(array([[1, 2],
               [2, 3],
               [3, 4]]))

        >>> Segment([[1,2],[2,3]]).concat([3, 4])
        Segment(array([[1, 2],
               [2, 3],
               [3, 4]]))
        """
        s = Segment(self)
        s.iconcat(*others)
        return s

    def iconcat(self, *others):
        """Concatenate segments (in place).

        Parameters
        ----------
        *others : segment arrays

        """
        self._data = segment_concatenate(self._data, *others)
        return self

    def __iadd__(self, value):
        """Concatenates segments or adds offset (in place).

        Parameters
        ----------
        value : Segment, or scalar or 1d array
            If value is a Segment object, then its segments are concatenated to this Segment. Otherwise, value is added
            as an offset to the segments.

        """
        if isinstance(value, Segment):
            return self.iconcat(value)

        return self.ioffset(value)

    def __add__(self, value):
        """Concatenate segments or add offset.

        Parameters
        ----------
        value : Segment, or scalar or 1d array
            If value is a Segment object, then a new Segment object with a concatenated list of segment is returned.
            Otherwise, value is added as an offset to the segments.

        Returns
        -------
        Segment
        """
        if isinstance(value, Segment):
            return self.concat(value)

        return self.offset(value)

    __radd__ = __add__

    def __sub__(self, value):
        """Subtract value.

        Parameters
        ----------
        value : scalar or 1d array

        Returns
        -------
        Segment
        """
        return self.offset(-value)

    __rsub__ = __sub__

    def __isub__(self, value):
        """Subtract value (in place).

        Parameter
        ---------
        value : scalar or 1d array

        """
        return self.ioffset(-value)

    __mul__ = scale
    __rmul__ = __mul__
    __imul__ = iscale

    def __truediv__(self, value):
        """Divide segment durations.

        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor.

        Returns
        -------
        Segment

        """
        return self.scale(1.0 / value)

    def __rtruediv__(self, value):
        return NotImplemented

    __div__ = __truediv__
    __rdiv__ = __rtruediv__

    def __itruediv__(self, value):
        """Divide segment durations (in place).

        Parameters
        ----------
        value : scalar or 1d array
            Scaling factor.

        """
        return self.iscale(1.0 / value)

    __idiv__ = __itruediv__

    def contains(self, value, issorted=True, expand=None):
        """Test if values are contained in segments.

        Segments are considered left closed and right open intervals.
        So, a value x is contained in a segment if start<=x and x<stop.

        Parameters
        ----------
        value : sorted 1d array
        issorted : bool
            Assumes vector x is sorted and will not sort it internally. Note that even if issorted is False,
            the third output argument will still return indices into the (internally) sorted vector.
        expand : bool
            Will expand the last output to full index arrays into 'x' for each segment. The default is True if issorted is False and
            vice versa. Note that for non-sorted data (issorted is False) and expand=False, the last output argument will
            contain start and stop indices into the (internally) sorted input array.

        Returns
        -------
        ndarray
            True for each value in x that is contained within any segment.
        ndarray
            For each segment the number of values in x that it contains.
        ndarray
            For each segment, the start and end indices of values in x that are contained within that segment.

        Examples
        --------

        >>> Segment([[1,2], [4,5]]).contains(4)
        (array([ True]), array([0, 1]), array([[-1, -1],
               [ 0,  0]]))

        >>> Segment([[1,2], [4,5]]).contains([0, 4])
        (array([False,  True]), array([0, 1]), array([[-1, -1],
               [ 1,  1]]))

        >>> Segment([[1,2], [4,5], [0,1], [3,5]]).contains([0, 4], expand=True)
        (array([ True,  True]), array([0, 1, 1, 1]), [array([-1]), array([1]), array([0]), array([1])])
        """
        # TODO: test if self is sorted
        # TODO: test if value is sorted
        # TODO: support scalars and nd-arrays for value?
        return segment_contains(self._data, value, issorted, expand)

    def __contains__(self, value):
        return self.contains(value)[0]

    def count(self, x):
        """Count number of segments.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
            For each value in x the number of segments that contain that value.

        Examples
        --------

        >>> Segment([[2,4], [2,6]]).count(3)
        array(2.)

        >>> Segment([[2,4], [2,6]]).count([3, 8])
        array([2., 0.])


        """
        return segment_count(self._data, x)

    def overlap(self, other=None):
        """Return absolute and relative overlaps between segments.

        Parameters
        ----------
        other : segment array, optional
            If other is not provided, then auto-overlaps are analyzed.

        Returns
        -------
        ndarray
            absolute overlap between all combinations of segments
        ndarray
            overlap relative to duration of first segment
        ndarray
            overlap relative to duration of second segment

        Examples
        --------

        >>> Segment([[2,4], [2,6]]).overlap()
        (array([[2., 2.],
               [2., 4.]]), array([[1. , 1. ],
               [0.5, 1. ]]), array([[1. , 0.5],
               [1. , 1. ]]))

        >>> Segment([[2,4], [2,6]]).overlap([5,8])
        (array([[0.],
               [1.]]), array([[0.  ],
               [0.25]]), array([[0.        ],
               [0.33333333]]))

        """
        return segment_overlap(self._data, other=other)

    def asindex(self, x, valid_only=False):
        """Convert segments to indices into vector.

        Parameters
        ----------
        x : ndarray
        valid_only : bool, optional
            set to True if invalid segments should be discarded otherwise invalid segment = [-1,-1]
        Returns
        -------
        Segment (indices)

        Examples
        --------

        >>> segment = Segment([[ 0 ,4],[ 5,12],[ 52,60]])
        >>> segment.asindex(np.linspace(0, 12, 48), valid_only=True)
        Segment(array([[ 0, 15],
               [20, 46]]))

        >>> segment = Segment([[ 0 ,4],[ 5,12],[ 52,60]])
        >>> segment.asindex(np.linspace(0, 12, 48))
        Segment(array([[ 0, 15],
               [20, 46],
               [-1, -1]]))
        """
        return Segment(segment_asindex(self._data, x, valid_only))

    def ijoin(self, gap=0):
        """Join segments with small inter-segment gap (in place).

        Parameters
        ----------
        gap : scalar
            Segments with an interval equal to or smaller than gap will be merged.

        """
        self._data = segment_join(self._data, gap=gap)
        return self

    def join(self, *args, **kwargs):
        """Join segments with small inter-segment gap (or overlapped).

        Parameters
        ----------
        gap : scalar
            Segments with an interval equal to or smaller than gap will be merged.

        Returns
        -------
        Segment

        Examples
        --------

        >>> Segment([[1,2],[2.5, 3.5], [5,6]]).join(gap=1)
        Segment(array([[1. , 3.5],
               [5. , 6. ]]))
        """
        s = Segment(self)
        s.ijoin(*args, **kwargs)
        return s

    def split(self, size=1, overlap=0, join=True, tol=1e-7):
        """Split segments into smaller segments with optional overlap.

        Parameters
        ----------
        size : scalar
            Duration of split segments.
        overlap : scalar
            Relative overlap (>=0. and <1.) between split segments.
        join : bool
            Join all split segments into a single segment array. If join is False, a list is returned with split segments
            for each original segment separately.
        tol : scalar
            Tolerance for determining number of bins.

        Returns
        -------
        Segment or list of Segments

        Examples
        --------

        >>> Segment([1,3]).split(1)
        Segment(array([[1, 2],
               [2, 3]]))

        >>> Segment([1,3]).split(1, overlap=0.25)
        Segment(array([[1.  , 2.  ],
               [1.75, 2.75]]))

        >>> Segment([[1,3], [2,4]]).split(1, join=False)
        [Segment(array([[1, 2],
               [2, 3]])), Segment(array([[2, 3],
               [3, 4]]))]
        """
        seg = segment_split(self._data, size=size, overlap=overlap, join=join, tol=tol)
        if len(seg) == 0:
            return Segment([])
        elif isinstance(seg, list):  # we have a list of segments
            return [Segment(x) for x in seg]
        else:
            return Segment(seg)

    def applyfcn(self, x, *args, **kwargs):
        """Apply function to segmented data.

        Parameters
        ----------
        x : ndarray
            The function is applied to values in this array that lie within
            the segments.
        separate : bool, opt, default=False
            Apply function to data in each segment separately
        function : callable, opt, default=len
            Function that takes one or more data arrays.
        default : any, opt, default=None
            Default value for segments that do not contain data (only used
            when separate is True)
        *args : ndarray-like
            Data arrays that are segmented (along first dimension) according
            to the corresponding values in x that lie within the segments,
            and passed to function.

        Returns
        -------
        ndarray or [ ndarray, ]
            Result of applying function to segmented data.
        """
        return segment_applyfcn(self._data, x, *args, **kwargs)

    def partition(self, **kwargs):
        """Partition segments into groups.

        Parameters
        ----------
        nparts : int
            Number of partitions
        method: 'block', 'random', 'sequence'
            Method of assigning segments to partitions.

        Returns
        -------
        Segment object
            partitioned subset of segments

        See Also
        --------
        fklab.general.partitions
        fklab.general.partition_vector

        """
        return partition_vector(self, **kwargs)
        # kwargs['size'] = self._data.shape[0]
        # return (self[idx] for idx in partitions( **kwargs ))

    def uniform_random(self, size=(1,)):
        """Sample values uniformly from segments.

        Parameters
        ----------
        size : tuple of ints
            Shape of returned array.

        Returns
        -------
        ndarray

        """
        return segment_uniform_random(self._data, size=size)
