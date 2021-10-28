import unittest

import numpy as np

import fklab.statistics.core


class TestStatisticBinned(unittest.TestCase):
    def test_bin_array_1d_usecase(self):
        bin_number = 21
        position = np.random.uniform(0, 100, size=(1000,))
        bins = np.linspace(0, 100, bin_number)
        index, valid, bin_edges, shape = fklab.statistics.core.bin_array(
            position, bins=bins
        )
        self.assertEqual(shape[0], bin_edges[0].size - 1)
        self.assertEqual(shape[0], bin_number - 1)

    def test_bin_array_2d_usecase(self):
        # generate data
        bin_number = 21

        position = np.random.uniform(-50, 50, size=(1000, 2))

        bins = np.linspace(-50, 50, bin_number)
        index, valid, bin_edges, shape = fklab.statistics.core.bin_array(
            position, bins=bins
        )

        self.assertEqual(len(shape), len(bin_edges))
        self.assertEqual(shape[0], bin_edges[0].size - 1)
        self.assertEqual(shape[1], bin_edges[1].size - 1)
        self.assertEqual(shape[1], bin_number - 1)

    def test_bin_array_exclude_data(self):
        bin_number = 21
        position = np.random.uniform(0, 100, size=(1000,))
        bins = np.linspace(0, 80, bin_number)
        index, valid, bin_edges, shape = fklab.statistics.core.bin_array(
            position, bins=bins
        )

        np.testing.assert_array_equal(valid, position < 80)

    def test_generate_full_binned_array(self):
        bin_number = 21
        position = np.random.uniform(-50, 50, size=(10, 2))
        bins = np.linspace(-50, 50, bin_number)
        speed = np.random.normal(50 - np.abs(position - 50), 5)
        index, valid, bin_edges, shape = fklab.statistics.core.bin_array(
            position, bins=bins
        )
        mu, groups = fklab.statistics.core.generate_full_binned_array(
            index[valid], speed[valid], fcn=np.mean
        )

        self.assertEqual(len(mu), 10)


if __name__ == "__main__":
    unittest.main()
