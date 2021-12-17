import doctest
import unittest

import numpy as np

import fklab.segments
from fklab.segments import Segment
from fklab.segments import SegmentError


class TestBasicAlgorithm(unittest.TestCase):
    def setUp(self):  # Add here any steps you want alway run before 1 test
        self.vector = np.linspace(0, 12, 48)
        self.s1 = [[0, 10], [20, 30], [50, 100]]
        self.s2 = [[4, 6], [15, 25], [40, 200]]
        self.s3 = [[8, 20], [35, 60], [150, 180]]

    def tearDown(self):
        del self.s1
        del self.s2
        del self.s3

    def test_check_segment(self):
        issegment = fklab.segments.check_segments([[0, 4], [5, 12]])
        np.testing.assert_array_equal(issegment, np.array([[0, 4], [5, 12]]))

    def test_check_not_segment(self):
        self.assertRaises(
            ValueError, lambda: fklab.segments.check_segments([["test", 4], [5, 12]])
        )

    def test_as_index(self):
        segment = Segment([[0, 4], [5, 12]])
        expected = [[0, 15], [20, 46]]
        result = segment.asindex(self.vector)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_as_index_full(self):
        segment = Segment([[0, 4], [4, 12]])
        expected = [[0, 15], [16, 46]]
        result = segment.asindex(self.vector)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_issegment(self):
        self.assertTrue(Segment.issegment(Segment([[0, 10], [20, 30], [50, 100]])))

    def test_segment_construct_from_list_no_copy(self):
        s1 = Segment(self.s1)
        s = Segment(s1, copy=False)
        self.assertIsInstance(s, Segment)
        self.assertTrue(s == s1)

    def test_segment_construct_empty(self):
        s = Segment([])
        self.assertIsInstance(s, Segment)

    def test_sort_segment_empty(self):
        s = Segment([])
        self.assertEqual(s.sort(), Segment([]))

    def test_segment_iteration(self):
        s1 = Segment([[1, 2], [3, 4]])
        for index, seq in enumerate(s1):
            if index == 0:
                np.testing.assert_array_equal(seq, np.array([1, 2]))
            if index == 1:
                np.testing.assert_array_equal(seq, np.array([3, 4]))

    def test_segment_equal(self):
        s1 = Segment([[1, 2], [3, 4], [10, 20]])
        s2 = Segment([[1, 2], [3, 4], [10, 20]])
        self.assertIsInstance(s1, type(s2))
        self.assertTrue(s1 == s2)
        self.assertFalse(s1 == [[1, 1], [2, 4], [10, 20]])
        self.assertFalse(s1 == [[1, 1], [2, 4]])

    def test_segment_equal_2(self):
        self.assertRaises(ValueError, lambda: Segment([[1, 2], [10, 20]]) == "a")
        self.assertRaises(ValueError, lambda: Segment([[1, 2], [10, 20]]) != "a")

    def test_segment_inequal(self):
        s1 = Segment([[1, 2], [3, 4], [10, 20]])
        s2 = Segment([[1, 2], [3, 4], [10, 20]])

        self.assertFalse(s1 != s2)
        self.assertTrue(s1 != [[1, 1], [2, 4]])
        self.assertTrue(s1 != [[1, 1], [2, 4], [10, 20]])

    def test_get_set_item(self):
        s1 = Segment([[1, 2], [3, 4], [10, 20]])
        s1[0] = [2, 3]
        self.assertEqual(s1[0], [2, 3])
        del s1[0]
        self.assertEqual(s1, Segment([[3, 4], [10, 20]]))

    def test_segment_exclusive(self):
        s1 = Segment([[0, 10], [20, 30], [50, 100]])
        s2 = Segment([[4, 6], [15, 25], [40, 200]])
        s1 = s1.exclusive(s2)
        np.testing.assert_array_equal(s1._data, np.array([[0, 4], [6, 10], [25, 30]]))

    def test_segment_exclusive_2(self):
        s1 = Segment([[0, 10], [20, 30], [50, 100]])
        s2 = Segment([[4, 6], [15, 25], [40, 200]])
        s3 = Segment([[8, 20], [35, 60], [150, 180]])
        self.assertEqual(s1 & ~s2 & ~s3, s1.exclusive(s2, s3))

    def test_segment_union(self):
        s1 = Segment([[0, 10], [20, 30], [50, 100]])
        s2 = Segment([[4, 6], [15, 25], [40, 200]])
        s1 = s1.union(s2)
        np.testing.assert_array_equal(
            s1._data, np.array([[0, 10], [15, 30], [40, 200]])
        )

    def test_segment_union_2(self):
        s1 = Segment([[0, 10], [20, 30], [50, 100]])
        s2 = Segment([[4, 6], [15, 25], [40, 200]])
        s3 = Segment([[8, 20], [35, 60], [150, 180]])
        self.assertEqual(s1 | s2 | s3, s1.union(s2, s3))

    def test_segment_intersection(self):
        s1 = Segment([[0, 10], [20, 30], [50, 100]])
        s2 = Segment([[4, 6], [15, 25], [40, 200]])
        s1 = s1.intersection(s2)
        np.testing.assert_array_equal(s1._data, np.array([[4, 6], [20, 25], [50, 100]]))

    def test_segment_intersection_2(self):
        s1 = Segment([[0, 10], [20, 30], [50, 100]])
        s2 = Segment([[4, 6], [15, 25], [40, 200]])
        s3 = Segment([[8, 20], [35, 60], [150, 180]])
        self.assertEqual(s1 & s2 & s3, s1.intersection(s2, s3))

    def test_segment_difference(self):
        s1 = Segment([[0, 10], [20, 30], [50, 100]])
        s2 = Segment([[4, 6], [15, 25], [40, 200]])
        s1 = s1.difference(s2)

        np.testing.assert_array_equal(
            s1._data,
            np.array([[0, 4], [6, 10], [15, 20], [25, 30], [40, 50], [100, 200]]),
        )


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(fklab.segments.segment))
    return tests


if __name__ == "__main__":
    unittest.main()
