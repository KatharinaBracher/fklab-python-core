import unittest

import numpy as np

from fklab.signals.core import remove_artefacts


class TestBasicAlgorithm(unittest.TestCase):
    def setUp(self):  # Add here any steps you want alway run before 1 test
        self.signal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.time = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.artefacts = [7, 8]

    def tearDown(self):  # Add here any steps you want alway run after 1 test
        del self.signal
        del self.time
        del self.artefacts

    def test_remove_artefacts(self):
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = remove_artefacts(
            self.signal, self.artefacts
        )  # default interpolation is linear
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
