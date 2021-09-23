import doctest
import unittest

import numpy as np

import fklab.segments.segment
from fklab.segments import Segment as seg


class TestBasicAlgorithm(unittest.TestCase):
    def setUp(self):  # Add here any steps you want alway run before 1 test
        self.vector = np.linspace(0, 12, 48)

    def test_as_index(self):
        segment = seg([[0, 4], [5, 12]])
        expected = [[0, 15], [20, 46]]
        result = segment.asindex(self.vector)
        np.testing.assert_array_equal(result.asarray(), expected)

    def test_as_index_full(self):
        segment = seg([[0, 4], [4, 12]])
        expected = [[0, 15], [16, 46]]
        result = segment.asindex(self.vector)
        np.testing.assert_array_equal(result.asarray(), expected)


def load_tests(loader, tests, ignore):
    tests.addTests(
        doctest.DocTestSuite(fklab.segments.segment)
    )  # example: fklab.events.event
    return tests


if __name__ == "__main__":
    unittest.main()
