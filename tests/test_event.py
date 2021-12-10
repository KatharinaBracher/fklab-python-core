import doctest
import unittest

import numpy as np

import fklab.events.event
from fklab.events.basic_algorithms import event_bin
from fklab.events.basic_algorithms import event_bursts
from fklab.events.basic_algorithms import event_rate
from fklab.events.basic_algorithms import filter_bursts
from fklab.events.basic_algorithms import filter_bursts_length
from fklab.events.basic_algorithms import filter_intervals
from fklab.events.basic_algorithms import split_eventstrings
from fklab.events.event import Event


class TestBasicAlgorithm(unittest.TestCase):
    def test_split_event_string_list(self):
        res = split_eventstrings([2, 3, 4, 6], ["a", "b", "a", "b"])
        self.assertTrue(list(res.keys()) == ["a", "b"])
        np.testing.assert_array_equal(res["a"], np.array([2, 4]))
        np.testing.assert_array_equal(res["b"], np.array([3, 6]))

    def test_split_event_string_less_event(self):
        self.assertRaises(
            Exception, lambda: split_eventstrings([2, 3, 4, 6], ["a", "b", "b"])
        )
        self.assertRaises(
            Exception,
            lambda: split_eventstrings([2, 3, 4, 6], ["a", "b", "b", "b", "c"]),
        )

    def test_check_event_rate(self):
        res = event_rate(np.array([1, 1.5, 2, 2.5]))
        self.assertEqual(res, 2.0)

    def test_check_events_rate(self):
        res = event_rate([np.array([1, 2, 3, 4]), np.array([2, 3, 4, 5])])
        self.assertEqual(list(res), [1, 1])

    def test_check_event_rate_sep_segments(self):
        res = event_rate(
            np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]),
            segments=[[1, 3], [3, 7]],
            separate=True,
        )
        np.testing.assert_array_equal(res, np.array([[2.0, 1.0]]))

    def test_event_bin_rate(self):
        res = event_bin(
            np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]),
            np.array([[3.0, 7], [1.0, 3]]),
            kind="rate",
        )
        np.testing.assert_array_equal(res, np.array([[1.0], [2.0]]))

    def test_event_bin_count(self):
        res = event_bin(
            np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 6.5, 7]),
            np.array([[1.0, 3], [2.0, 7]]),
            kind="count",
        )
        np.testing.assert_array_equal(res, np.array([[4], [7]]))

    def test_event_bin_binary(self):
        res = event_bin(
            np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 6.5, 7]),
            np.array([[0.0, 1.0], [2.0, 7]]),
            kind="binary",
        )
        np.testing.assert_array_equal(res, np.array([[0], [1]]))

    def test_event_bin_raise(self):
        self.assertRaises(
            Exception,
            lambda: event_bin(
                np.array([1, 1.5]), np.array([[0.0, 1.0], [2.0, 7]]), kind=""
            ),
        )

    def test_event_burst_marks(self):
        np.warnings.filterwarnings("ignore")
        np.testing.assert_array_equal(
            event_bursts(Event([1, 2, 3]), marks=[1, 2, 3]), np.array([0, 0, 0])
        )
        self.assertRaises(
            Exception, lambda: event_bursts(Event([1, 2, 3]), marks=[1, 2])
        )

    def test_filter_bursts_reduce(self):
        np.warnings.filterwarnings("ignore")
        event, idx = filter_bursts(
            np.array([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]),
            method="reduce",
            intervals=[0.9, 1.1],
        )
        np.testing.assert_array_equal(
            idx, np.array([True, False, True, True, False, False, False, True, True])
        )

    def test_filter_bursts_remove(self):
        np.warnings.filterwarnings("ignore")
        event, idx = filter_bursts(
            np.array([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]),
            method="remove",
            intervals=[0.9, 1.1],
        )
        np.testing.assert_array_equal(
            idx, np.array([False, False, True, False, False, False, False, True, True])
        )

    def test_filter_bursts_isolate(self):
        np.warnings.filterwarnings("ignore")
        event, idx = filter_bursts(
            np.array([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]),
            method="isolate",
            intervals=[0.9, 1.1],
        )
        np.testing.assert_array_equal(
            idx, np.array([True, True, False, True, True, True, True, False, False])
        )

    def test_filter_bursts_isolatereduce(self):
        np.warnings.filterwarnings("ignore")
        event, idx = filter_bursts(
            np.array([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]),
            method="isolatereduce",
            intervals=[0.9, 1.1],
        )
        np.testing.assert_array_equal(
            idx, np.array([True, False, False, True, False, False, False, False, False])
        )

    def test_filter_length_burst(self):
        burst = filter_bursts_length(
            np.array([1.0, 3.0, 0.0, 1.0, 2.0, 2.0, 3.0, 0.0, 0.0])
        )

        np.testing.assert_array_equal(
            burst, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

    def test_filter_length_burst_ranged(self):
        burst = filter_bursts_length(
            np.array([1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0]), nevents=[3, 4]
        )
        np.testing.assert_array_equal(
            burst, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

    def test_event_interval(self):
        res, index = filter_intervals(
            np.array([1, 2, 2.5, 3, 4, 5, 6, 6.5, 7]), mininterval=1
        )
        np.testing.assert_array_equal(index, [2, 3, 7, 8])


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(fklab.events.event))
    tests.addTests(doctest.DocTestSuite(fklab.events.basic_algorithms))
    return tests


if __name__ == "__main__":

    unittest.main()
