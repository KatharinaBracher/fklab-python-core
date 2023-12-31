import unittest

import numpy as np

from fklab.signals.core import remove_artefacts


class TestBasicAlgorithm(unittest.TestCase):
    def setUp(self):  # Add here any steps you want alway run before 1 test
        self.signal = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.time = [0, 1, 2, 3, 4, 4.5, 5, 6, 7, 8]
        self.artefacts = [2, 3]

    def tearDown(self):  # Add here any steps you want alway run after 1 test
        del self.signal
        del self.time
        del self.artefacts

    def test_remove_artefacts_fill(self):
        expected = np.array([0, 0, 0, 0, 4, 5, 6, 7, 8, 9])
        result = remove_artefacts(
            self.signal, self.artefacts, time=self.time, interp="fill", fill_value=0
        )  # default interpolation is linear
        np.testing.assert_array_equal(result, expected)

    def test_remove_artefacts_fill_nan(self):
        expected = np.array([0, np.nan, np.nan, np.nan, 4, 5, 6, 7, 8, 9])
        result = remove_artefacts(
            self.signal,
            self.artefacts,
            time=self.time,
            interp="fill",
            fill_value=np.nan,
        )  # default interpolation is linear
        np.testing.assert_array_equal(result, expected)

    def test_remove_artefacts_interpolate_linear(self):
        signal = np.array([0, 1, 1.2, 1.3, 4, 5, 6, 7, 8, 9])
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = remove_artefacts(
            signal, self.artefacts, time=self.time
        )  # default interpolation is linear
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
