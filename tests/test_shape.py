import unittest

import numpy as np

from fklab.geometry.shapes import ellipse
from fklab.geometry.shapes import rectangle


class BoxedCase(unittest.TestCase):
    def test_is_circle_1(self):
        shape = ellipse(center=[0, 0], size=[10, 10])
        self.assertEqual(shape.iscircle, True)

    def test_is_circle_2(self):
        shape = ellipse(center=[0, 0], size=10)
        self.assertEqual(shape.iscircle, True)

    def test_is_circle_3(self):
        shape = ellipse(center=[0, 0], size=[10, 20], orientation=0)
        self.assertEqual(shape.iscircle, False)

    def test_is_square_1(self):
        shape = rectangle(center=[0, 0], size=[10, 10])
        self.assertEqual(shape.issquare, True)

    def test_is_square_2(self):
        shape = rectangle(center=[0, 0], size=10)
        self.assertEqual(shape.issquare, True)

    def test_is_square_3(self):
        shape = rectangle(center=[0, 0], size=[10, 20])
        self.assertEqual(shape.issquare, False)

    def test_eccentricity_ellipse(self):
        shape = ellipse(center=[0, 0], size=[10, 20], orientation=0)
        self.assertAlmostEqual(shape.eccentricity, 0.86, places=1)

    def test_eccentricity_circle(self):
        shape = ellipse(center=[0, 0], size=10, orientation=0)
        self.assertEqual(shape.eccentricity, 0)

    def test_boundingbox_circle(self):
        shape = ellipse(center=[0, 0], size=10, orientation=0)
        box = rectangle(center=[0, 0], size=[10, 10])
        self.assertAlmostEqual(shape.boundingbox.size, box.size)

    def test_boundingbox_ellipse(self):
        shape = ellipse(center=[0, 0], size=[1, 4], orientation=0)
        box = rectangle(center=[0, 0], size=[1, 4])
        self.assertEqual(shape.boundingbox.size[0], box.size[0])
        self.assertEqual(shape.boundingbox.size[1], box.size[1])

    def test_boundingbox_ellipse_inclined(self):
        shape = ellipse(center=[0, 0], size=[1, 2], orientation=np.pi / 4)
        box = rectangle(center=[0, 0], size=1.58)
        self.assertAlmostEqual(shape.boundingbox.size[0], box.size[0], places=1)

    def test_boundingbox_rectangle_inclined(self):
        shape = rectangle(center=[0, 0], size=[1, 2], orientation=np.pi / 4)
        box = rectangle(center=[0, 0], size=2.12)
        self.assertAlmostEqual(shape.boundingbox.size[0], box.size[0], places=1)

    def test_contains_ellipse_1(self):
        shape = ellipse(center=[0, 0], size=[1, 2])
        self.assertEqual(shape.contains([0, 1]), True)

    def test_contains_cercle_1(self):
        shape = ellipse(center=[0, 0], size=[2, 2])
        self.assertEqual(shape.contains([0, 0.9]), True)

    def test_contains_ellipse_2(self):
        shape = ellipse(center=[0, 0], size=[1, 2])
        self.assertEqual(shape.contains([0, 2])[0], False)

    def test_contains_cercle_2(self):
        shape = ellipse(center=[0, 0], size=[1, 1])
        self.assertEqual(shape.contains([0.5, 3])[0], False)

    def test_contains_rectangle_1(self):
        shape = rectangle(center=[0, 0], size=[1, 2])
        self.assertEqual(shape.contains([0, 0.9]), True)

    def test_contains_rectangle_2(self):
        shape = rectangle(center=[0, 0], size=[1, 2])
        self.assertEqual(shape.contains([0.75, 0.75]), False)


if __name__ == "__main__":
    unittest.main()
