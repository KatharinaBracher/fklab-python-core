"""
Classes for working with geometrical shapes, including methods for
projecting 2D points to the shape outline (e.g. as used for behavioral
track linearization).


"""
import abc
import collections
import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.spatial
import scipy.special

import fklab.statistics.circular as circ
import fklab.utilities.yaml as yaml
from . import transforms as tf
from . import utilities as util
from fklab.version._core_version._version import __version__

__all__ = [
    "ellipse",
    "rectangle",
    "polyline",
    "polygon",
    "graph",
    "multishape",
    "ngon",
    "triangle",
    "pentagon",
    "hexagon",
]

# class hierarchy
# shape <- path <- solid <- boxed <- (ellipse,rectangle)
# shape <- path <- (polyline, graph)
# shape <- path <- (polyline,solid) <- polygon
# shape <- multishape

# we need to create a single common meta base class
class meta(yaml.YAMLObjectMetaclass, abc.ABCMeta):
    pass


class shape(yaml.YAMLObject, metaclass=meta):
    def __init__(self, name=""):
        self.name = name

    def to_dict(self):
        return collections.OrderedDict(name=self.name)

    @abc.abstractmethod
    def __repr__(self):
        return "abstract shape"

    @property
    def name(self):
        """Name of shape."""
        return self._name

    @name.setter
    def name(self, value):
        if value is None or len(value) == 0:
            value = "shape_{}_{:05d}".format(
                datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"),
                np.random.randint(100000),
            )
        else:
            value = str(value)
        self._name = value

    @property
    @abc.abstractmethod
    def center(self):
        """Return center location of shape."""
        return np.zeros(2)

    @abc.abstractmethod
    def scale(self, factor, origin=None):
        """Scale shape.

        Parameters
        ----------
            factor : scalar or [factor_x, factor_y]
            origin : (x, y), optional

        """
        pass

    @abc.abstractmethod
    def rotate(self, angle, origin=None):
        """Rotate shape around its center.

        Parameters
        ----------
        angle : scalar
        origin : (x, y), optional

        """
        pass

    @abc.abstractmethod
    def translate(self, offset):
        """Translate shape.

        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]

        """
        pass

    @abc.abstractmethod
    def transform(self, tform):
        """Transform shape.

        Parameters
        ----------
        tform : Transform

        """
        pass

    @property
    @abc.abstractmethod
    def boundingbox(self):
        """Bounding box of shape."""
        pass

    @property
    def ispath(self):
        """Test if shape is a path."""
        return isinstance(self, path)

    @property
    def issolid(self):
        """Test if shape is a solid."""
        return isinstance(self, solid)


class path(shape):
    def __init__(self, forced_length=None, **kwargs):
        super().__init__(**kwargs)
        self._forced_length = (
            float(forced_length) if not forced_length is None else None
        )

    def to_dict(self):
        d = super().to_dict()
        d["forced_length"] = self.forced_length
        return d

    @property
    def forced_length(self):
        return self._forced_length

    @property
    @abc.abstractmethod
    def pathlength(self):
        """Return the path length or circumference of shape."""
        return 0

    @abc.abstractmethod
    def point2path(self, points):
        """Project points to shape path."""
        pass

    @abc.abstractmethod
    def path2point(self, points):
        """Unproject points on shape path."""
        pass

    @abc.abstractmethod
    def path2edge(self, x):
        """Convert between path and edge representations."""
        pass

    @abc.abstractmethod
    def edge2path(self, x):
        """Convert between edge and path representations."""
        pass

    @abc.abstractmethod
    def tangent(self, x):
        """Compute path tangent at given locations along path."""
        pass

    def normal(self, x):
        """Compute path normal at given locations along path."""
        return self.tangent(x) - 0.5 * np.pi

    def gradient(self, x, dx=1):
        """Compute gradient at given locations along path."""
        # convert to dist_along_path if given as (edge,dist_along_edge) tuple
        if isinstance(x, tuple):
            x = self.edge2path(x)

        return np.gradient(x, dx)

    @abc.abstractmethod
    def random_on(self, n):
        """Draw random coordinates on path."""
        pass

    def random_along(self, n=1):
        """Draw uniform random locations along path."""
        return np.random.uniform(low=0, high=self.pathlength, size=n)

    def distance(self, x, y):
        """Compute distance along path.

        Parameters
        ----------
        x,y : array or tuple
            distance along path or (edge index, distance along edge) tuple

        Returns
        -------
        ndarray
            element-wise distance between `x` and `y`

        """
        if isinstance(x, tuple):
            x = self.edge2path(x)
        if isinstance(y, tuple):
            y = self.edge2path(y)

        return y - x

    @abc.abstractmethod
    def samplepath(self, oversampling=None):
        """Densely sample points along path."""
        pass

    def bin(self, binsize, **kwargs):
        """Divide path into bins.

        Parameters
        ----------
        binsize : float
            Desired bin size

        Returns
        -------
        bins : 1d array
            bin edges
        nbins : int
            number of bins
        binsize : float
            actual bin size

        """
        return _bin_path([self.pathlength], binsize)

    def path2map(self, resolution=1.0, expand=1.4):
        """Construct a map of distances along/to a path.

        Parameters
        ----------
        resolution : scalar
            size of each point in the map
        expand : scalar
            Factor by which to scale the bounding box of the track

        Returns
        -------
        L, D : 2d array
            Maps of linearized position along the track (L) and distance to
            the track (D) for every point in the map.
        x, y : 1d array
            Coordinates for each point in the map.

        """
        nx, ny = np.ceil(self.boundingbox.size[[0, -1]] * 1.4 / resolution)
        cx, cy = self.boundingbox.center

        vx = (np.arange(nx) - (nx - 1) / 2) * resolution + cx
        vy = (np.arange(ny) - (ny - 1) / 2) * resolution + cy
        mx, my = np.meshgrid(vx, vy)

        L, D, _, _ = self.point2path(np.column_stack([mx.ravel(), my.ravel()]))
        L = np.reshape(L, mx.shape)
        D = np.reshape(D, mx.shape)

        return L, D, (vx, vy)

    def plot_path(self, axes=None, **kwargs):
        """Plot the path.

        Parameters
        ----------
        axes : Axes
        **kwargs :
            Extra keyword arguments for `plot` function

        Returns
        -------
        Axes

        """
        if axes is None:
            axes = plt.gca()

        xy = self.samplepath()
        axes.plot(xy[:, 0], xy[:, 1], **kwargs)

        return axes


class solid(path):
    @property
    @abc.abstractmethod
    def area(self):
        """Return area of closed shape."""
        return 0

    @abc.abstractmethod
    def contains(self, points):
        """Test if points are contained within shape."""
        return False

    @abc.abstractmethod
    def random_in(self, n):
        """Draw random points inside shape."""
        pass

    def gradient(self, x, dx=1):
        """Compute gradient at given locations along path.

        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple
        dx : scalar

        Returns
        -------
        ndarray

        """
        # convert linear position to (pseudo) angle and unwrap
        f = 2 * np.pi / self.pathlength
        x = np.unwrap(f * x) / f
        # compute gradient
        g = np.gradient(x, dx)
        return g

    def distance(self, x, y):
        """Compute distance along path.

        Parameters
        ----------
        x,y : array or tuple
              distance along path or (edge index, distance along edge) tuple

        Returns
        -------
        ndarray
            element-wise distance between `x` and `y`

        """
        return circ.diff(x, y, low=0.0, high=self.pathlength, directed=True)


class boxed(solid):
    def __init__(self, center=None, size=1, orientation=0, **kwargs):
        super(boxed, self).__init__(**kwargs)

        if center is None:
            center = [0, 0]

        self.center = center

        self.size = size
        self.orientation = orientation

    def to_dict(self):
        d = super().to_dict()
        d["center"] = self.center.tolist()
        d["size"] = self.size.tolist()
        d["orientation"] = float(self.orientation)
        return d

    @property
    def center(self):
        """Center of shape."""
        return self._center

    @center.setter
    def center(self, value):
        try:
            value = np.array(value, dtype=np.float64)
            value = value.reshape((2,))
        except:
            raise TypeError("Invalid value for center.")

        self._center = value

    @property
    def size(self):
        """Size of shape."""
        return self._size

    @size.setter
    def size(self, value):
        try:
            value = np.array(value, dtype=np.float64)
            value = value.reshape((value.size,))
        except:
            raise TypeError("Invalid value for size.")

        if value.size < 1 or value.size > 2:
            raise TypeError("Expecting scalar, or length-2 array for setting size.")

        if value.size == 2 and value[0] == value[1]:
            value = value[0:1]

        self._size = value

    @property
    def orientation(self):
        """Orientation of shape."""
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        try:
            value = np.array(value, dtype=np.float64)
            value = value.reshape((1,))
        except:
            raise TypeError("Invalid value for orientation.")

        value = np.mod(value, 2 * np.pi)

        self._orientation = value

    def scale(self, factor, origin=None):
        """Scale shape.

        Parameters
        ----------
            factor : scalar or [factor_x, factor_y]
            origin : (x, y), optional

        """
        factor = np.array(factor, dtype=np.float64)
        self.size = self._size * factor

        if not origin is None:
            origin = np.array(origin, dtype=np.float64)
            self.center = (self.center - origin) * factor + origin

    def rotate(self, angle, origin=None):
        """Rotate shape around its center.

        Parameters
        ----------
        angle : scalar
        origin : (x, y), optional

        """
        angle = np.array(angle, dtype=np.float64)
        self.orientation = self._orientation + angle

        if not origin is None:
            self.center = tf.Rotate(angle, origin).transform(self.center)

    def translate(self, offset):
        """Translate shape.

        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]

        """
        offset = np.array(offset, dtype=np.float64)
        self.center = self._center + offset

    def boxvertices(self):
        """Vertices of enclosing box."""
        v = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        tform = (
            tf.Scale(self.size)
            + tf.Rotate(self.orientation)
            + tf.Translate(self.center)
        )
        v = tform.transform(v)
        return v


class ellipse(boxed):
    r"""Ellipse shape.

    Parameters
    ----------
    center : (2,) array_like
        x and y coordinates of ellipse center

    size : scalar or (2,) array_like, opt
        length of ellipse major and minor axes
        if radius option is specified, radius should not be specified as size = 2*radius

    radius : scalar or (2,) array_like, opt
        length of semi-ellipse major and semi-minor axes
        if size option is specified, radius should not be specified as size = 2*radius

    orientation : float, optional
        orientation of the ellipse in radians (default is 0)
    name : str, optional
        name of the shape (default is "")


    .. note:: An ellipse is represented by the equations (canonical form):

        .. math::

            x = a*cos(t) \\
            y = b*sin(t)

        The starting point of an ellipse is at (a,0). The angles increase
        counter-clockwise. At the starting point, the tangent angle is
        1.5 :math:`\pi` and the normal is 0.


    """

    _inv_elliptic = None  # cached elleptic integral interpolator

    yaml_tag = "!ellipse_shape"

    def __init__(
        self, center=(0, 0), size=None, radius=None, orientation=0.0, **kwargs
    ):

        if size and radius:
            assert np.array(size) / 2 == np.array(
                radius
            ), "radius and size options cannot be specified in the same time except if size = 2*radius"

        if not size:
            if radius:
                size = np.array(radius) * 2
            else:
                size = 1

        super().__init__(center=center, size=size, orientation=orientation, **kwargs)

    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent ellipse as YAML."""
        d = data.to_dict()
        node = dumper.represent_mapping(cls.yaml_tag, iter(d.items()))
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct ellipse from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(**d)

    def __repr__(self):
        if self.iscircle:
            return "circle (radius={radius[0]}, center=[{center[0]},{center[1]}])".format(
                center=self.center, radius=self.size
            )
        else:
            return "ellipse (size=[{size[0]},{size[1]}], center=[{center[0]},{center[1]}], orientation={orientation[0]} rad)".format(
                size=self.size, center=self.center, orientation=self.orientation
            )

    def _get_inv_elliptic(self):
        if self._inv_elliptic is None or np.any(self._inv_elliptic[0] != self._size):
            # update cached inverse elliptic integral interpolator
            self._inv_elliptic = (
                self._size.copy(),
                _inv_ellipeinc(self._size[0], self._size[1]),
            )

        return self._inv_elliptic[1]

    @property
    def radius(self):
        return self._size / 2

    @property
    def iscircle(self):
        """Verify if shape is a circle (i.e. major and minor axes are equal)."""
        return self._size.size == 1

    @property
    def eccentricity(self):
        """Return eccentricity of ellipse.

        The eccentricity can be thought of as a measure of how much an
        ellipse deviates from being circular. A circle has eccentricity
        of 0 and an ellipse has eccentricity between 0 and 1.
        """
        if self.iscircle:
            return 0
        else:
            a = np.max(self._size)
            b = np.min(self._size)
            return np.sqrt(1 - (b / a) ** 2)

    def transform(self, tform):
        """Apply transformation to ellipse.

        The ellipse is first approximated by a polygon before the
        transformation is applied.

        Parameters
        ----------
        tform : Transform
            the transformation to apply

        Returns
        -------
        polygon

        """
        p = self.aspolygon()
        p.transform(tform)
        return p

    @property
    def boundingbox(self):
        """Axis aligned bounding box of ellipse."""
        if self.iscircle:
            return rectangle(center=self._center, size=np.ones(2) * 2 * self.radius)
        else:
            # parameterized equations for ellipse:
            # x = center[0] + size[0]*cos(t)*cos(orientation) - size[1]*sin(t)*sin(orientation)
            # y = center[1] + size[1]*sin(t)*cos(orientation) + size[0]*cos(t)*sin(orientation)
            # then solve dx/dt=0 and dy/dt=0
            # resulting in (for x): t = atan( -size[1]*tan(orientation)/size[0] ) + n*PI
            # and (for y): t = atan( size[1]*cot(orientation)/size[0] ) + n*PI
            # plugging in `t` into parameterized equations will give the extreme
            # x and y coordinates of the ellipse (and thus of the bounding box)
            t = np.arctan(-self.radius[1] * np.tan(self._orientation) / self.radius[0])
            x = (
                self._center[0]
                + self.radius[0] * np.cos([t, t - np.pi]) * np.cos(self._orientation)
                - self.radius[1] * np.sin([t, t - np.pi]) * np.sin(self._orientation)
            )
            if self.orientation == 0:
                t = np.pi / 2
            else:
                t = np.arctan(
                    self.radius[1] / (np.tan(self._orientation) * self.radius[0])
                )
            y = (
                self._center[1]
                + self.radius[1] * np.sin([t, t - np.pi]) * np.cos(self._orientation)
                + self.radius[0] * np.cos([t, t - np.pi]) * np.sin(self._orientation)
            )
            return rectangle(
                center=[(x[0] + x[1]) / 2, (y[0] + y[1]) / 2],
                size=np.abs([float(x[1] - x[0]), float(y[1] - y[0])]),
                orientation=0,
            )

    @property
    def pathlength(self):
        """Circumference of ellipse."""
        if not self.forced_length is None:
            L = self.forced_length
        else:
            L = self._compute_pathlength()

        return L

    def _compute_pathlength(self):

        if self.iscircle:
            L = 2 * np.pi * self.radius
        else:
            L = 4 * np.max(self.radius) * sp.special.ellipe(self.eccentricity ** 2)

        return L

    def point2path(self, points):  # OK
        """Project points to ellipse circumference.

        Parameters
        ----------
        points : (n,2) array

        Returns
        -------
        dist_along_path : array
            distance along the circumference to the projected point
        dist_to_path: array
            distance between the original point and the project point on
            the circumference
        point_on_path: array
            coordinates of projected point on circumference
        edge: tuple
            representation of projected point as edge index and distance
            along edge. An ellipse consists of only a single edge with
            index 0.

        Notes
        -----
        Ellipses (but not circles) are first approximated by a polygon.

        """
        if self.iscircle:
            [theta, rho] = util.cart2pol(
                points[:, 0] - self._center[0], points[:, 1] - self._center[1]
            )
            dist_along_path = circ.wrap(theta - self.orientation) * self.radius
            dist_to_path = rho - self.radius
            point_on_path = np.vstack(
                (
                    self.radius * np.cos(theta) + self.center[0],
                    self.radius * np.sin(theta) + self.center[1],
                )
            ).T
        else:
            # approximate for ellipse
            p = self.aspolygon(oversampling=100)
            (dist_along_path, dist_to_path, point_on_path) = p.point2path(points)[0:3]
            # correct linear distance
            dist_along_path = dist_along_path * self.pathlength / p.pathlength

        if not self.forced_length is None:
            dist_along_path = (
                dist_along_path * self.forced_length / self._compute_pathlength()
            )

        return (
            dist_along_path,
            dist_to_path,
            point_on_path,
            (np.zeros(dist_along_path.shape, dtype=np.int), dist_along_path),
        )

    def path2point(self, x, distance=0, _method=1):  # OK
        """Convert points along circumference to 2d coordinates.

        Parameters
        ----------
        x : array or tuple
            distance along circumference or ( edge index, distance along edge )
            representation
        distance : float, optional
            distance to offset point from circumference (default is 0)

        Returns
        -------
        xy : (n,2) array
            x,y coordinates

        """
        # convert to distance along path, if given as (edge,dist_along_edge) tuple
        if isinstance(x, tuple):
            x = x[1]

        if not self.forced_length is None:
            x = x * self._compute_pathlength / self.forced_length

        if self.iscircle:
            [x, y] = util.pol2cart(
                x / self.radius + self.orientation, self.radius + distance
            )
            xy = np.vstack((x + self.center[0], y + self.center[1])).T
            return xy
        elif _method == 1:
            # transform x to ellipse parameter t
            fcn = self._get_inv_elliptic()
            t = fcn(x)
            # compute x and y coordinates
            xy = np.vstack(
                (
                    (self.radius[0] + distance) * np.cos(t),
                    (self.radius[1] + distance) * np.sin(t),
                )
            ).T
            tform = tf.Rotate(self.orientation) + tf.Translate(self.center)
            xy = tform.transform(xy)
            return xy
        else:
            p = self.aspolygon(oversampling=100)
            x = x * p.pathlength / self.pathlength
            return p.path2point(x, distance)

    def path2edge(self, x):
        """Convert path to edge representation.

        Parameters
        ----------
        x : array
            distance along ellipse circumference

        Returns
        -------
        tuple
            representation of point on circumference as edge index and
            distance along edge. An ellipse consists of only a single
            edge with index 0.

        """
        return (np.zeros(x.shape, dtype=np.int), x)

    def edge2path(self, x):
        """Convert edge to path representation.

        Parameters
        ----------
        x : tuple
            (edge index, distance along edge) representation of points on
            ellipse circumference

        Returns
        -------
        array
            distance along ellipse circumference

        """
        if not isinstance(x, tuple):
            raise ValueError(
                "Expecting tuple (edge index, distance along edge) representation of points on ellipse circumference"
            )
        return x[1]

    def tangent(self, x):  # OK
        """Compute tangential angle at points along path.

        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple

        Returns
        -------
        array
            tangential angle in radians

        """
        # convert to distance along path, if given as (edge,dist_along_edge) tuple
        if isinstance(x, tuple):
            x = x[1]

        if not self.forced_length is None:
            x = x * self._compute_pathlength / self.forced_length

        if self.iscircle:
            d = circ.wrap(x / self.radius + 0.5 * np.pi + self.orientation)
        else:
            # transform to ellipse parameter t
            fcn = self._get_inv_elliptic()
            t = fcn(x)
            # compute tangent angle
            d = circ.wrap(
                np.arctan2(self.radius[1] * np.cos(t), -np.sin(t) * self.radius[0])
                + self.orientation
            )

        return d

    def samplepath(self, oversampling=20, openpath=False):  # OK
        """Sample regurlarly points on  circumference.

        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along circumference. If oversampling is 1, four points are
            sampled. In general, 4*oversampling points are sampled
            (default is 20)
        openpath : bool, optional
            Whether or not to leave sampled path open. If False, the
            first and last sampled point will be identical
            (default is False).

        Returns
        -------
        array
            x,y coordinates of sampled points

        """
        npoints = oversampling * 4

        if not openpath:
            npoints += 1

        vertices = np.empty((npoints, 2), dtype=np.float64)
        vertices[:, 0], vertices[:, 1] = util.pol2cart(
            np.pi * 2 * np.linspace(0, 1, num=npoints, endpoint=not openpath), 1.0
        )

        t = (
            tf.Scale(self.radius)
            + tf.Rotate(self._orientation)
            + tf.Translate(self._center)
        )
        vertices = t.transform(vertices)

        return vertices

    def random_on(self, n):  # OK
        """Sample random points on circumference.

        Parameters
        ----------
        n : int
            number of randomly sampled points

        Returns
        -------
        array
            x,y coordinates of randomly sampled points

        """
        if self.iscircle:
            # generate random points on circle
            x, y = util.pol2cart(
                np.random.uniform(low=0, high=2 * np.pi, size=n), self.radius[0]
            )
            xy = np.vstack((x, y)).T
        else:
            # generate uniform random arc lengths
            L = np.random.uniform(low=0, high=self._compute_pathlength(), size=n)
            # transform to ellipse parameter t
            fcn = self._get_inv_elliptic()
            t = fcn(L)
            # compute x and y coordinates
            xy = np.vstack((self.radius[0] * np.cos(t), self.radius[1] * np.sin(t))).T

        # transform points
        tform = tf.Rotate(self._orientation) + tf.Translate(self._center)
        xy = tform.transform(xy)

        return xy

    @property
    def area(self):
        """Surface area of ellipse."""
        return np.pi * self.radius[0] * self.radius[-1]

    def contains(self, points):
        """Test if points are contained within ellipse.

        Parameters
        ----------
        points : array_like
            x,y coordinates of points to be tested.

        Returns
        -------
        bool array
            True if point is contained within ellipse.

        """
        points = np.array(points)
        if points.ndim == 1 and points.size == 2:
            points = points.reshape((1, 2))

        assert points.ndim == 2 and points.shape[1] == 2

        if self.iscircle:
            return np.sum((points - self._center) ** 2, axis=1) <= self.radius ** 2
        else:
            # transform points: translate(-center), rotate(-orientation), scale(1/size)
            # test if distance of point to (0,0) <= 1
            t = (
                tf.Translate(-self._center)
                + tf.Rotate(-self._orientation)
                + tf.Scale(1 / self.radius)
            )
            points = t.transform(points)
            return np.sum(points ** 2, axis=1) <= 1

    def random_in(self, n):
        """Sample random points inside ellipse.

        Parameters
        ----------
        n : int
            number of randomly sampled points

        Returns
        -------
        array
            x,y coordinates of randomly sampled points

        """
        # generate random points in unit circle
        x, y = util.pol2cart(
            np.random.uniform(low=0, high=2 * np.pi, size=n),
            np.sqrt(np.random.uniform(low=0, high=1, size=n)),
        )
        # transform points
        t = (
            tf.Scale(self.radius)
            + tf.Rotate(self._orientation)
            + tf.Translate(self._center)
        )
        xy = t.transform(np.vstack((x, y)).T)
        return xy

    def aspolygon(self, oversampling=20):
        """Convert ellipse to polygon.

        Parameters
        ----------
        oversampling : int

        Returns
        -------
        polygon

        """
        vertices = self.samplepath(oversampling=oversampling, openpath=True)
        return polygon(
            vertices=vertices,
            spline=False,
            name=self.name,
            forced_length=self.forced_length,
        )


class rectangle(boxed):
    r"""Rectangle shape.

    Parameters
    ----------
    center : (2,) array_like
        x and y coordinates of rectangle center
    size : scalar or (2,) array_like
        major and minor axes of rectangle
    orientation : float, optional
        orientation of the rectangle in radians (default is 0)
    name : str, optional
        name of the shape (default is "")

    Attributes
    ----------
    name
    center
    size
    orientation
    boundingbox
    ispath
    issolid
    issquare
    pathlength
    area

    """

    yaml_tag = "!rectangle_shape"

    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent rectangle as YAML."""
        d = data.to_dict()
        node = dumper.represent_mapping(cls.yaml_tag, iter(d.items()))
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct rectangle from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(**d)

    def __repr__(self):
        if self.issquare:
            return "square (size={size[0]}, center=[{center[0]},{center[1]}], orientation={orientation[0]} rad)".format(
                size=self.size, center=self.center, orientation=self.orientation
            )
        else:
            return "rectangle (size=[{size[0]},{size[1]}], center=[{center[0]},{center[1]}], orientation={orientation[0]} rad)".format(
                size=self.size, center=self.center, orientation=self.orientation
            )

    @property
    def issquare(self):
        """Verify if shape is a square (i.e. major and minor axes are equal)."""
        return len(self._size) == 1

    def transform(self, tform):
        """Apply transformation to rectangle.

        The rectangle is first approximated by a polygon before the
        transformation is applied.

        Parameters
        ----------
        tform : Transform
            the transformation to apply

        Returns
        -------
        polygon

        """
        p = self.aspolygon()
        p.transform(tform)
        return p

    @property
    def boundingbox(self):
        """Axis aligned bounding box of rectangle."""
        vertices = self.boxvertices()
        maxval = np.max(vertices, axis=0)
        minval = np.min(vertices, axis=0)

        return rectangle(
            center=(maxval + minval) / 2.0, size=maxval - minval, orientation=0
        )

    @property
    def pathlength(self):
        """Perimeter of rectangle."""
        if not self.forced_length is None:
            L = self.forced_length
        else:
            L = self._compute_pathlength()

        return L

    def _compute_pathlength(self):
        return 2 * (self._size[0] + self._size[-1])

    def point2path(self, *args, **kwargs):
        """Project points to rectangle perimeter.

        Parameters
        ----------
        points : (n,2) array

        Returns
        -------
        dist_along_path : array
            distance along the perimeter to the projected point
        dist_to_path: array
            distance between the original point and the projected point on
            the perimeter
        point_on_path: array
            coordinates of projected point on perimeter
        edge: tuple
            representation of projected point as edge index and distance
            along edge. A rectangle consists of four edges.

        """
        return self.aspolygon().point2path(*args, **kwargs)

    def path2point(self, *args, **kwargs):
        """Convert points along perimeter to 2d coordinates.

        Parameters
        ----------
        x : array or tuple
            distance along perimeter or ( edge index, distance along edge )
            representation
        distance : float, optional
            distance to offset point from perimeter (default is 0)

        Returns
        -------
        xy : (n,2) array
            x,y coordinates

        """
        return self.aspolygon().path2point(*args, **kwargs)

    def edge2path(self, *args, **kwargs):
        """Convert edge to path representation.

        Parameters
        ----------
        x : tuple
            (edge index, distance along edge) representation of points on
            rectangle perimeter

        Returns
        -------
        array
            distance along rectangle perimeter

        """
        return self.aspolygon().edge2path(*args, **kwargs)

    def path2edge(self, *args, **kwargs):
        """Convert path to edge representation.

        Parameters
        ----------
        x : array
            distance along rectangle perimeter

        Returns
        -------
        tuple
            representation of point on perimeter as edge index and
            distance along edge.

        """
        return self.aspolygon().path2edge(*args, **kwargs)

    def tangent(self, *args, **kwargs):
        """Compute tangential angle at points along path.

        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple

        Returns
        -------
        array
            tangential angle in radians

        """
        return self.aspolygon().tangent(*args, **kwargs)

    def samplepath(self, oversampling=1, openpath=False):
        """Sample regurlarly points on a perimeter.

        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along circumference. If oversampling is 1, four points are
            sampled. In general, 4*oversampling points are sampled
            (default is 1)
        openpath : bool, optional
            Whether or not to leave sampled path open. If False, the
            first and last sampled point will be identical
            (default is False).

        Returns
        -------
        array
            x,y coordinates of sampled points

        """
        npoints = np.floor(oversampling) * 4

        if not openpath:
            npoints += 1

        # construct vertices, the first edge is the right-most edge
        vertices = np.array(
            [[0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]
        )
        fcn = sp.interpolate.interp1d(
            np.arange(5), vertices, kind="linear", axis=0, bounds_error=False
        )

        x = np.linspace(0, 4, num=np.int(npoints), endpoint=not openpath)

        vertices = fcn(x)

        t = (
            tf.Scale(self._size)
            + tf.Rotate(self._orientation)
            + tf.Translate(self._center)
        )
        vertices = t.transform(vertices)

        return vertices

    def random_on(self, n):
        """Sample random points on perimeter.

        Parameters
        ----------
        n : int
            number of randomly sampled points

        Returns
        -------
        array
            x,y coordinates of randomly sampled points

        """
        ratio = self._size[0] / np.sum(self._size)
        nx = int(np.round(ratio * n))

        xy = np.random.uniform(low=-0.5, high=0.5, size=(n, 2))

        xy[0:nx, 1] = np.random.randint(2, size=nx) - 0.5
        xy[nx:, 0] = np.random.randint(2, size=int(n - nx)) - 0.5

        t = (
            tf.Scale(self._size)
            + tf.Rotate(self._orientation)
            + tf.Translate(self._center)
        )
        xy = t.transform(xy)

        return xy

    @property
    def area(self):
        """Surface area of rectangle."""
        return self._size[0] * self._size[-1]

    def contains(self, *args, **kwargs):
        """Test if points are contained within rectangle.

        Parameters
        ----------
        points : array_like
            x,y coordinates of points to be tested.

        Returns
        -------
        bool array
            True if point is contained within rectangle.

        """
        return self.aspolygon().contains(*args, **kwargs)

    def random_in(self, n):
        """Sample random points inside rectangle.

        Parameters
        ----------
        n : int
            number of randomly sampled points

        Returns
        -------
        array
            x,y coordinates of randomly sampled points

        """
        # create random points in unit square
        xy = np.random.uniform(low=-0.5, high=0.5, size=(n, 2))
        t = (
            tf.Scale(self._size)
            + tf.Rotate(self._orientation)
            + tf.Translate(self._center)
        )
        xy = t.transform(xy)
        return xy

    def aspolygon(self, oversampling=1):
        """Convert rectangle to polygon.

        Parameters
        ----------
        oversampling : int

        Returns
        -------
        polygon

        """
        vertices = self.samplepath(oversampling=oversampling, openpath=True)
        return polygon(
            vertices=vertices,
            spline=False,
            name=self.name,
            forced_length=self.forced_length,
        )


class polyline(path):
    """Polyline shape.

    Parameters
    ----------
    vertices : (n,2) array
    spline : bool
        Apply spline interpolation.
    name : str, optional
        name of the shape (default is "")

    Attributes
    ----------
    name
    center
    boundingbox
    numvertices
    vertices
    isspline
    ispath
    issolid
    pathlength
    edgelengths

    """

    _length = None  # cached path length
    _isspline = False  # True if interpolated spline
    _vertices = []  # (n,2) array of vertices
    _edge_lengths = []
    _path_integral = []

    yaml_tag = "!polyline_shape"

    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent polyline as YAML."""
        d = data.to_dict()
        node = dumper.represent_mapping(cls.yaml_tag, iter(d.items()))
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct polyline from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(**d)

    def __init__(self, vertices=(), spline=False, **kwargs):
        super(polyline, self).__init__(**kwargs)
        self.vertices = vertices
        self.isspline = spline

        # to record times of transformation after last cache updated.
        self._modifications = 0

    def to_dict(self):
        d = super().to_dict()
        d["vertices"] = self.vertices.tolist()
        d["spline"] = self.isspline
        return d

    def __repr__(self):
        return "{shapetype} {klass} with {n} vertices".format(
            shapetype="spline" if self.isspline else "straight",
            klass=self.__class__.__name__,
            n=self.numvertices,
        )

    @property
    def numvertices(self):
        """Number of polyline vertices."""
        return self._vertices.shape[0]

    @property
    def center(self):
        """Center of shape."""
        return np.mean(self._vertices, axis=0)

    def _get_expanded_vertices(self, spline=False):
        if spline:
            return self._sampled_spline[0]
        elif self.issolid:
            return np.concatenate((self._vertices, self._vertices[0:1, :]), axis=0)
        else:
            return self._vertices

    def _update_cached_values(self, force=False):
        if not force and self._modifications == 0:
            return

        vertices = self._get_expanded_vertices(spline=False)

        if self.isspline:
            self._sampled_spline, self._spline = _sample_spline(
                vertices, oversampling=50, closed=self.issolid, openpath=False
            )
            L = np.sqrt(
                np.sum(np.diff(self._sampled_spline[0], n=1, axis=0) ** 2, axis=1)
            )
            pathlength = np.sum(L)
        else:
            try:
                del self._sampled_spline
                del self._spline
            except:
                pass

            if self.numvertices < 2:
                pathlength = 0
                self.edge_lengths = np.array([])
                L = 0
            else:
                L = np.sqrt(np.sum(np.diff(vertices, n=1, axis=0) ** 2, axis=1))
                pathlength = np.sum(L)

        self._length = pathlength
        self._path_integral = np.cumsum(np.concatenate(([0], L)))
        self._path_integral[-1] = pathlength
        self._modifications = 0  # reset counter

        if self.isspline:
            self._edge_lengths = np.diff(
                sp.interpolate.interp1d(
                    self._sampled_spline[1], self._path_integral, kind="linear"
                )(self._spline[1])
            )
        else:
            self._edge_lengths = L

    @property
    def vertices(self):
        """Polyline vertices."""
        return self._vertices

    @vertices.setter
    def vertices(self, value):
        value = np.array(value, dtype=np.float64)

        if value.size == 0:
            value = np.zeros((0, 2))
        elif value.ndim != 2 or value.shape[1] != 2:
            raise TypeError("Vertices should have shape (n,2).")

        self._vertices = value
        self._update_cached_values(True)

    @property
    def isspline(self):
        """Whether spline interpolation is enabled."""
        return self._isspline and self.numvertices > 3

    @isspline.setter
    def isspline(self, value):
        self._isspline = bool(value)
        self._update_cached_values(True)

    def scale(self, factor, origin=None):
        """Scale shape.

        Parameters
        ----------
            factor : scalar or [factor_x, factor_y]
            origin : (x, y), optional

        """
        if origin is None:
            origin = self.center

        t = tf.Scale(factor=factor, origin=origin)
        self._vertices = t.transform(self._vertices)
        self._modifications += 1

    def rotate(self, angle, origin=None):
        """Rotate shape around its center.

        Parameters
        ----------
        angle : scalar
        origin : (x, y), optional

        """
        if origin is None:
            origin = self.center

        t = tf.Rotate(angle=angle, origin=origin)
        self._vertices = t.transform(self._vertices)
        self._modifications += 1

    def translate(self, offset):
        """Translate shape.

        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]

        """
        t = tf.Translate(offset=offset)
        self._vertices = t.transform(self._vertices)
        self._modifications += 1

    def transform(self, tform):
        """Transform shape.

        Parameters
        ----------
        tform : Transform

        """
        self._vertices = tform.transform(self._vertices)
        self._modifications += 1

    @property
    def boundingbox(self):
        """Axis-aligned bounding box of polyline."""
        vertices = self._get_expanded_vertices(self.isspline)
        maxval = np.max(vertices, axis=0)
        minval = np.min(vertices, axis=0)
        return rectangle(
            center=(maxval + minval) / 2.0, size=maxval - minval, orientation=0
        )

    @property
    def pathlength(self):
        """Path length of polyline."""
        if self.forced_length is None:
            self._update_cached_values()
            return self._length
        else:
            return self.forced_length

    def point2path(self, points, clip=("normal", "normal")):  # OK
        """Project points to polyline.

        Parameters
        ----------
        points : (n,2) array
        clip : 2-element tuple
            Clipping behavior for the two polyline end points.
            'normal' = points beyond line segment are projected to the line end points
            'blunt' = points beyond line segment are excluded
            'full' = points are projected to the full line that passes through the end points

        Returns
        -------
        dist_along_path : array
            distance along the path to the projected point
        dist_to_path: array
            distance between the original point and the projected point on
            the path
        point_on_path: array
            coordinates of projected point on path
        edge: tuple
            representation of projected point as edge index and distance
            along edge.

        """
        self._update_cached_values()

        vertices = self._get_expanded_vertices(self.isspline)
        (
            dist_to_path,
            point_on_path,
            edge,
            dist_along_edge,
            dist_along_path,
        ) = util.point2polyline(vertices, points, clip)

        if not self.forced_length is None:
            dist_along_path = dist_along_path * self.forced_length / self._length
            dist_along_edge = dist_along_edge * self.forced_length / self._length

        return (dist_along_path, dist_to_path, point_on_path, (edge, dist_along_edge))

    def path2point(self, x, distance=0):  # OK
        """Convert points along path to 2d coordinates.

        Parameters
        ----------
        x : array or tuple
            distance along path or ( edge index, distance along edge )
            representation
        distance : float, optional
            distance to offset point from path (default is 0)

        Returns
        -------
        xy : (n,2) array
            x,y coordinates

        """
        self._update_cached_values()

        if isinstance(x, tuple):
            x = self.edge2path(x)

        if not self.forced_length is None:
            x = x * self._length / self.forced_length

        vertices = self._get_expanded_vertices(spline=self.isspline)
        L = self._path_integral

        xy = scipy.interpolate.interp1d(L, vertices, kind="linear", axis=0)(
            x
        )  # missing: extrapolation!

        distance = np.asarray(distance)

        if np.any(distance != 0):
            normal = self.normal(x)
            xy[:, 0] += distance * np.cos(normal)
            xy[:, 1] += distance * np.sin(normal)

        return xy

    def path2edge(self, x):
        """Convert path to edge representation.

        Parameters
        ----------
        x : array
            distance along path

        Returns
        -------
        tuple
            representation of point on path as edge index and
            distance along edge.

        """
        return _path2edge(x, self.edgelengths, self.pathlength)

    def edge2path(self, x):
        """Convert edge to path representation.

        Parameters
        ----------
        x : tuple
            (edge index, distance along edge) representation of points on
            path

        Returns
        -------
        array
            distance along path

        """
        return _edge2path(x, self.edgelengths, self.pathlength)

    @property
    def edgelengths(self):
        """Lengths of polyline edges."""
        self._update_cached_values()

        if self.forced_length is None:
            return self._edge_lengths
        else:
            return self._edge_lengths * self.forced_length / self._length

    def tangent(self, x):  # OK?
        """Compute tangential angle at points along path.

        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple

        Returns
        -------
        array
            tangential angle in radians

        """
        if self.isspline:
            self._update_cached_values()

            if isinstance(x, tuple):
                x = self.edge2path(x)

            if not self.forced_length is None:
                x = x * self._length / self.forced_length

            dx, dy = sp.interpolate.splev(
                sp.interpolate.interp1d(
                    self._path_integral, self._sampled_spline[1], kind="linear", axis=0
                )(x),
                self._spline[0],
                der=1,
            )
            d = np.arctan2(dy, dx)
        else:
            if not isinstance(x, tuple):
                x = self.path2edge(x)[0]

            vertices = self._get_expanded_vertices()
            d = np.arctan2(
                vertices[x + 1, 1] - vertices[x, 1], vertices[x + 1, 0] - vertices[x, 0]
            )

        return d

    def samplepath(self, oversampling=None, openpath=False):  # OK
        """Sample regurlarly points on the path.

        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along path (default is 1; with spline interpolation default is 20)
        openpath : bool, optional
            Whether or not to leave sampled path open. If False, the
            first and last sampled point will be identical
            (default is False).

        Returns
        -------
        array
            x,y coordinates of sampled points

        """
        if oversampling is None:
            oversampling = 20 if self.isspline else 1

        oversampling = np.floor(oversampling)
        vertices = self._get_expanded_vertices()

        if oversampling == 1:
            return vertices.copy()

        if self.isspline:
            sampled_spline, spline = _sample_spline(
                vertices,
                oversampling=oversampling,
                closed=self.issolid,
                openpath=openpath,
            )
            vertices = sampled_spline[0]
        else:
            vertices = _sample_polyline(
                vertices,
                oversampling=oversampling,
                closed=self.issolid,
                openpath=openpath,
            )

        return vertices

    def random_on(self, n):  # OK?
        """Sample random points on path.

        Parameters
        ----------
        n : int
            number of randomly sampled points

        Returns
        -------
        array
            x,y coordinates of randomly sampled points

        """
        # draw points uniformly from [0,L), where L is length of polyline
        # map points back to 2D

        p = self.random_along(n)
        xy = self.path2point(p)

        return xy

    def aspolygon(self):
        """Convert polyline to closed polygon."""
        return polygon(
            vertices=self._vertices,
            spline=self.isspline,
            name=self.name,
            forced_length=self.forced_length,
        )


class polygon(polyline, solid):
    """Polygon shape.

    Parameters
    ----------
    vertices :
    spline : bool
        Apply spline interpolation.
    name : str, optional
        name of the shape (default is "")

    Attributes
    ----------
    numvertices
    area
    center
    vertices
    isspline
    boundingbox
    pathlength
    edgelengths

    """

    yaml_tag = "!polygon_shape"

    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent polygon as YAML."""
        d = data.to_dict()
        node = dumper.represent_mapping(cls.yaml_tag, iter(d.items()))
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct polygon from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(**d)

    def __init__(self, vertices=(), spline=False, **kwargs):
        super(polyline, self).__init__(**kwargs)
        self.vertices = vertices
        self.isspline = spline

    @property
    def area(self):
        """Polygon area."""
        vertices = self._get_expanded_vertices(self.isspline)
        return np.abs(util.polygonarea(vertices))

    def contains(self, points):
        """Test if points are contained within polygon.

        Parameters
        ----------
        points : array_like
            x,y coordinates of points to be tested.

        Returns
        -------
        bool array
            True if point is contained within polygon.

        """
        if isinstance(points, list):
            points = np.array(points)
        if len(points.shape) == 1:
            points = points.reshape((1, 2))

        vertices = self._get_expanded_vertices(self.isspline)
        return util.inpoly(vertices, points)

    def random_in(self, n=1):
        """Sample random points inside polygon.

        Parameters
        ----------
        n : int
            number of randomly sampled points

        Returns
        -------
        array
            x,y coordinates of randomly sampled points

        """
        # use rejection sampling
        # first get the bounding box
        # compute fraction of bounding box area occupied by polygon (won't work for complex polygons (i.e. with self intersections), which should not be allowed anyays)
        # randomly sample points from bounding box
        # reject all points that are not withon polygon
        # continue sampling until required number of points have been found

        if not self.isclosed:
            raise TypeError("Polygon needs to be closed.")

        # a rejection sampling approach is used to generate random points
        # first let's get the bounding box (used to generate random points)
        bb = self.boundingbox()
        # based on the area of the polygon, relative to its bounding box,
        # we determine the oversampling fraction
        f = 1.02 * bb.area / self.area

        # pre-allocate the return array
        xy = np.zeros((n, 2), dtype=np.float64)

        npoints = (
            0
        )  # number of random points inside polygon that have been generated so far
        niteration = 0  # number of iterations
        MAXITER = 10  # maximum number of iterations - we should never reach this

        while niteration < MAXITER:
            # determine the number of random points to generate
            nsamples = np.ceil((n - npoints) * f)
            # draw random points from bounding box
            samples = bb.randompoints(nsamples)
            # test if points are inside polygon
            b = self.contains(samples)
            nvalid = np.sum(b)

            if (
                npoints + nvalid
            ) > n:  # we have generated more valid random points than we needed
                xy[npoints:n, :] = samples[b][0 : (n - npoints)]
                npoints = n
                break
            else:  # we have generated fewer random points than we needed, so let's save them and continue
                xy[npoints : (npoints + nvalid), :] = samples[b]

            npoints = npoints + nvalid  # update the number of generated random points
            niteration += 1

        assert npoints == n

        return xy

    def aspolyline(self):
        """Convert polygon to open polyline."""
        return polyline(
            vertices=self._vertices,
            spline=self.isspline,
            name=self.name,
            forced_length=self.forced_length,
        )


class graph(path):
    """Graph of nodes connected by polylines.

    Parameters
    ----------
    polylines : sequence of polylines
    nodes : (n,2) array
    name : str, optional
        name of the shape (default is "")


    """

    yaml_tag = "!graph_shape"

    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent graph as YAML."""
        d = data.to_dict()
        node = dumper.represent_mapping(cls.yaml_tag, iter(d.items()))
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct graph from YAML."""
        d = loader.construct_mapping(node, deep=True)

        return cls(**d)

    def __init__(self, polylines=(), nodes=None, **kwargs):
        super(graph, self).__init__(**kwargs)

        nodes, polylines, edges = _check_graph(nodes, polylines)

        # check if polylines have unique names
        edge_names = [p.name for p in polylines]
        if len(edge_names) != len(set(edge_names)):
            raise ValueError("Polyline edges do not have unique names.")

        numnodes = nodes.shape[0]

        # make sure all polylines and nodes are connected
        if np.any(edges < 0) or not np.all(np.in1d(edges, np.arange(numnodes))):
            raise ValueError("Polylines and nodes are not all connected.")

        if np.any(edges[:, 0] == edges[:, 1]):
            raise ValueError("Polylines cannot start and end at the same vertex.")

        if numnodes < 2 or len(polylines) == 0:
            raise ValueError("Need at least two vertices and 1 polyline")

        self._graph = np.zeros((numnodes, numnodes), dtype=np.float64) + np.inf
        self._graph[np.diag_indices(numnodes)] = 0

        factor = 1
        if not self.forced_length is None:
            factor = self.forced_length / np.sum([p.pathlength for p in polylines])

        for index, p in enumerate(polylines):
            self._graph[edges[index, 0], edges[index, 1]] = np.minimum(
                self._graph[edges[index, 0], edges[index, 1]], factor * p.pathlength
            )
            self._graph[edges[index, 1], edges[index, 0]] = self._graph[
                edges[index, 0], edges[index, 1]
            ]

        self._edges = edges
        self._nodes = nodes
        self._polylines = polylines

        self._pathlengths, _ = util.floyd_warshall(self._graph)

    @property
    def edge_names(self):
        return [p.name for p in self._polylines]

    def to_dict(self):
        d = super().to_dict()
        d["nodes"] = self._nodes.tolist()
        d["polylines"] = self._polylines
        return d

    def __repr__(self):
        return "graph with {n} nodes and {ne} edges".format(
            n=self.numnodes, ne=self.numedges
        )

    @property
    def center(self):
        """Center of graph."""
        # center of mass of all nodes
        return np.mean(self._nodes, axis=0)

    @property
    def pathlength(self):
        """Total length of all paths in graph."""
        return np.sum(self.edgelengths)

    @property
    def edgelengths(self):
        """Path lengths of graph edges."""
        L = np.array([p.pathlength for p in self._polylines])
        if not self.forced_length is None:
            L = L * self.forced_length / np.sum(L)

        return L

    @property
    def numnodes(self):
        """Number of nodes in graph."""
        return self._nodes.shape[0]

    @property
    def numedges(self):
        """Number of edges in graph."""
        return len(self._polylines)

    def scale(self, factor, origin=None):
        """Scale shape.

        Parameters
        ----------
            factor : scalar or [factor_x, factor_y]
            origin : (x, y), optional

        """
        if origin is None:
            origin = self.center

        t = tf.Scale(factor=factor, origin=origin)
        self._nodes = t.transform(self._nodes)
        for p in self._polylines:
            p.transform(t)

    def rotate(self, angle, origin=None):
        """Rotate shape around its center.

        Parameters
        ----------
        angle : scalar
        origin : (x, y), optional

        """
        if origin is None:
            origin = self.center

        t = tf.Rotate(angle=angle, origin=origin)
        self._nodes = t.transform(self._nodes)
        for p in self._polylines:
            p.transform(t)

    def translate(self, offset):
        """Translate shape.

        Parameters
        ----------
        offset : scalar or [offset_x, offset_y]

        """
        t = tf.Translate(offset=offset)
        self._nodes = t.transform(self._nodes)
        for p in self._polylines:
            p.transform(t)

    def transform(self, tform):
        """Transform shape.

        Parameters
        ----------
        tform : Transform

        """
        self._nodes = tform.transform(self._nodes)
        for p in self._polylines:
            p.transform(tform)

    def path2edge(self, x):
        """Convert path to edge representation.

        Parameters
        ----------
        x : array
            distance along path

        Returns
        -------
        tuple
            representation of point on path as edge index and
            distance along edge.

        """
        return _path2edge(x, self.edgelengths, self.pathlength)

    def edge2path(self, x):
        """Convert edge to path representation.

        Parameters
        ----------
        x : tuple
            (edge index, distance along edge) representation of points on
            path

        Returns
        -------
        array
            distance along path

        """
        return _edge2path(x, self.edgelengths, self.pathlength)

    def point2path(self, points):  # OK?
        """Project points to graph.

        Parameters
        ----------
        points : (n,2) array

        Returns
        -------
        dist_along_path : array
            distance along the path to the projected point
        dist_to_path: array
            distance between the original point and the projected point on
            the path
        point_on_path: array
            coordinates of projected point on path
        edge: tuple
            representation of projected point as edge index and distance
            along edge.

        """
        points = np.asarray(points)

        # ignore warnings when comparing NaNs
        old_settings = np.seterr(invalid="ignore")

        edge = -np.ones(points.shape[0], dtype=np.int)
        dist_along_edge = np.full(points.shape[0], np.nan, dtype=np.float64)
        dist_to_path = np.full(points.shape[0], np.inf, dtype=np.float64)
        point_on_path = np.full(points.shape, np.nan, dtype=np.float64)

        # loop though all edges
        for k, p in enumerate(self._polylines):
            # compute point2path for edge
            (ld, d, pp) = p.point2path(points)[0:3]

            # test if dist_to_path is smallest so far
            idx = np.abs(d) < np.abs(dist_to_path)
            edge[idx] = k
            dist_along_edge[idx] = np.minimum(
                ld[idx], p.pathlength - 0.0001
            )  # exclude endpoints of segments
            dist_to_path[idx] = d[idx]
            point_on_path[idx] = pp[idx]

        valid = ~np.isnan(dist_along_edge)

        if not self.forced_length is None:
            dist_along_edge = (
                dist_along_edge
                * self.forced_length
                / np.sum([p.pathlength for p in self._polylines])
            )

        L = np.cumsum(np.concatenate(([0], self.edgelengths)))
        L[-1] = self.pathlength

        dist_along_path = np.empty(dist_along_edge.shape) * np.nan
        dist_along_path[valid] = dist_along_edge[valid] + L[edge[valid]]

        np.seterr(**old_settings)

        return (dist_along_path, dist_to_path, point_on_path, (edge, dist_along_edge))

    def path2point(self, x, distance=0):  # OK?
        """Convert points along path to 2d coordinates.

        Parameters
        ----------
        x : array or tuple
            distance along path or ( edge index, distance along edge )
            representation
        distance : float, optional
            distance to offset point from path (default is 0)

        Returns
        -------
        xy : (n,2) array
            x,y coordinates

        """
        if not isinstance(x, tuple):
            edge, dist_along_edge = self.path2edge(x)
        else:
            edge, dist_along_edge = x

        if not self.forced_length is None:
            dist_along_edge = (
                dist_along_edge
                * np.sum([p.pathlength for p in self._polylines])
                / self.forced_length
            )

        xy = np.empty((edge.size, 2)) * np.nan

        for k, p in enumerate(self._polylines):
            # find points on this edge
            idx = edge == k
            if np.any(idx):
                xy[idx] = p.path2point(dist_along_edge[idx], distance)

        return xy

    def tangent(self, x):  # OK?
        """Compute tangential angle at points along path.

        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple

        Returns
        -------
        array
            tangential angle in radians
        """
        if not isinstance(x, tuple):
            edge, dist_along_edge = self.path2edge(x)
        else:
            edge, dist_along_edge = x

        if not self.forced_length is None:
            dist_along_edge = (
                dist_along_edge
                * np.sum([p.pathlength for p in self._polylines])
                / self.forced_length
            )

        d = np.empty((edge.size)) * np.nan

        for k, p in enumerate(self._polylines):

            idx = edge == k
            if np.any(idx):
                d[idx] = p.tangent(dist_along_edge[idx])

        return d

    def gradient(self, x, dx=1):  # OK?
        """Compute gradient at given locations along path.

        Parameters
        ----------
        x : array or tuple
            distance along path or (edge index, distance along edge) tuple
        dx : scalar

        Returns
        -------
        ndarray

        """
        if isinstance(x, tuple):
            x = self.edge2path(x)

        g = np.empty(x.size) * np.nan
        g[0] = self.distance(x[0], x[1]) / dx
        g[-1] = self.distance(x[-2], x[-1]) / dx
        g[1:-1] = (self.distance(x[1:-1], x[2:]) - self.distance(x[1:-1], x[0:-2])) / (
            2 * dx
        )

        return g

    def distance(self, x, y):  # OK? - element-wise distance only
        """Compute distance along path.

        Parameters
        ----------
        x,y : array or tuple
            distance along path or (edge index, distance along edge) tuple

        Returns
        -------
        ndarray
            element-wise distance between `x` and `y`

        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if x.shape != y.shape:
            raise ValueError("x and y arrays need to have the same shape")

        if x.ndim > 1:
            original_shape = x.shape
            x = x.ravel()
            y = y.ravel()
        else:
            original_shape = None

        if not isinstance(x, tuple):
            x_edge, x_dist_along_path = self.path2edge(x)
        else:
            x_edge, x_dist_along_path = x

        if not isinstance(y, tuple):
            y_edge, y_dist_along_path = self.path2edge(y)
        else:
            y_edge, y_dist_along_path = y

        L = self.edgelengths

        d = np.full(x_edge.size, np.inf)

        # this short-cut does not produce correct results, because it assumes
        # that the shortest path between two points on the same edge is along that edge
        idx = x_edge == y_edge
        d[idx] = y_dist_along_path[idx] - x_dist_along_path[idx]
        # idx = np.nonzero(~idx)

        # TODO: deal with NaNs

        tmp = np.vstack(
            (
                -(
                    self._pathlengths[self._edges[x_edge, 0], self._edges[y_edge, 0]]
                    + x_dist_along_path
                    + y_dist_along_path
                ),
                -(
                    self._pathlengths[self._edges[x_edge, 0], self._edges[y_edge, 1]]
                    + x_dist_along_path
                    + L[y_edge]
                    - y_dist_along_path
                ),
                self._pathlengths[self._edges[x_edge, 1], self._edges[y_edge, 0]]
                + L[x_edge]
                - x_dist_along_path
                + y_dist_along_path,
                self._pathlengths[self._edges[x_edge, 1], self._edges[y_edge, 1]]
                + L[x_edge]
                - x_dist_along_path
                + L[y_edge]
                - y_dist_along_path,
                d,
            )
        )

        mi = np.argmin(np.abs(tmp), axis=0)
        d = tmp[mi, np.arange(tmp.shape[1])]

        if not original_shape is None:
            d = d.reshape(original_shape)

        return d

    def random_on(self, n):  # OK?
        """Sample random points on path.

        Parameters
        ----------
        n : int
            number of randomly sampled points

        Returns
        -------
        array
            x,y coordinates of randomly sampled points

        """
        # draw points uniformly from [0,L), where L is pathlength
        # map points back to 2D

        p = self.random_along(n)

        xy = self.path2point(p)

        return xy

    @property
    def boundingbox(self):  # OK?
        """Axis-aligned bounding box of graph."""
        xymax = np.zeros(2) - np.inf
        xymin = np.zeros(2) + np.inf

        for p in self._polylines:
            bb = p.boundingbox.boxvertices()
            xymax = np.maximum(np.max(bb, axis=0), xymax)
            xymin = np.minimum(np.min(bb, axis=0), xymin)

        return rectangle(
            center=(xymax + xymin) / 2.0, size=xymax - xymin, orientation=0
        )

    def samplepath(self, oversampling=None):  # TODO
        """Regular sampling of points on path.

        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along path.

        Returns
        -------
        array
            x,y coordinates of sampled points

        """
        return np.concatenate(
            [p.samplepath(oversampling=oversampling) for p in self._polylines], axis=0
        )

    def bin(self, binsize, separate=True):
        """Divide path into bins.

        Parameters
        ----------
        binsize : float
            Desired bin size
        separate : bool
            treat graph edges separately

        Returns
        -------
        bins : 1d array
            bin edges
        nbins : int
            number of bins
        binsize : float
            actual bin size

        """
        if not separate:
            return _bin_path([self.pathlength], binsize)
        else:
            return _bin_path(self.edgelengths, binsize)

    def plot_path(self, axes=None, **kwargs):
        """Plot the graph path.

        Parameters
        ----------
        axes : Axes
        **kwargs :
            Extra keyword arguments for `plot` function

        Returns
        -------
        Axes

        """
        if axes is None:
            axes = plt.gca()

        xy = [x.samplepath() for x in self._polylines]
        [axes.plot(x[:, 0], x[:, 1], **kwargs) for x in xy]

        return axes


class multishape(shape):
    """Collection of shapes.

    Parameters
    ----------
    *args : shape objects

    Attributes
    ----------
    name
    ispath
    issolid
    numshapes
    center
    boundingbox
    pathlength
    shapelengths

    """

    yaml_tag = "!multi_shape"

    @classmethod
    def to_yaml(cls, dumper, data):
        """Represent multishape as YAML."""
        d = dict(shapes=data._shapes)
        node = dumper.represent_mapping(cls.yaml_tag, iter(d.items()))
        return node

    @classmethod
    def from_yaml(cls, loader, node):
        """Construct multishape from YAML."""
        d = loader.construct_mapping(node, deep=True)
        return cls(*d["shapes"], name=d["name"])

    def __init__(self, *args, **kwargs):
        super(multishape, self).__init__(**kwargs)
        if any([not isinstance(x, shape) for x in args]):
            raise ValueError("Expecting shape objects.")

        # check if shapes have unique names
        names = [s.name for s in args]
        if len(names) != len(set(names)):
            raise ValueError("Shapes do not have unique names.")

        self._shapes = list(args)

    @property
    def shape_names(self):
        return [s.name for s in self._shapes]

    def __repr__(self):
        return "multishape with {n} shapes".format(n=len(self._shapes))

    @property
    def numshapes(self):
        """Number of shapes in collection."""
        return len(self._shapes)

    @property
    def center(self):
        """Averaged center of shapes in collection."""
        if self.numshapes == 0:
            return np.empty(2) * np.nan
        else:
            return np.mean(np.vstack([x.center for x in self._shapes]).T, axis=0)

    def scale(self, factor, origin=None):
        """Scaling transform is not implemented."""
        raise NotImplementedError()

    def rotate(self, angle, origin=None):
        """Rotation transform is not implemented."""
        raise NotImplementedError()

    def translate(self, offset):
        """Translate transform is not implemented."""
        raise NotImplementedError()

    def transform(self, tform):
        """Transformations are not implemented."""
        raise NotImplementedError()

    @property
    def boundingbox(self):
        """Axis-aligned bounding box of shapes in collection."""
        xymax = np.zeros(2) - np.inf
        xymin = np.zeros(2) + np.inf

        for x in self._shapes:
            bb = x.boundingbox.boxvertices()
            xymax = np.maximum(np.max(bb, axis=0), xymax)
            xymin = np.minimum(np.min(bb, axis=0), xymin)

        return rectangle(
            center=(xymax + xymin) / 2.0, size=xymax - xymin, orientation=0
        )

    @property
    def ispath(self):
        """Verify if any shape in collection is a path."""
        return any([x.ispath for x in self._shapes])

    @property
    def issolid(self):
        """Verify if any shape in collection is a solid."""
        return any([x.issolid for x in self._shapes])

    def point2path(self, points, context=None):
        """Project points to shape collection.

        Parameters
        ----------
        points : (n,2) array
        context : shape
            Restrict projection to one of the shapes.

        Returns
        -------
        dist_along_path : array
            distance along the path to the projected point
        dist_to_path: array
            distance between the original point and the projected point on
            the path
        point_on_path: array
            coordinates of projected point on path
        shape: tuple
            representation of projected point as shape index and distance
            along shape.
        """
        # TODO: if context is None, then for each point find nearest shape
        if context is None:
            raise NotImplementedError("Need a context.")

        # TODO: check correctness of code below. Is dist_along_path_expanded correct??

        # find index of shape context in self._shapes
        idx = self._shapes.index(context)
        dist_along_path, dist_to_path, point_on_path, dist_along_shape = self._shapes[
            idx
        ].point2path(points)

        if not self.forced_length is None:
            dist_along_path = (
                dist_along_path
                * self.forced_length
                / np.sum([x.pathlength for x in self._shapes])
            )

        dist_along_path_expanded = (
            np.zeros(dist_along_path.shape) + idx,
            dist_along_path,
        )
        dist_along_path = dist_along_path + np.sum(self.shapelengths[0:idx])
        return (dist_along_path, dist_to_path, point_on_path, dist_along_path_expanded)

    def path2point(self, x):
        """Convert points along path to 2d coordinates.

        Parameters
        ----------
        x : array or tuple
            distance along path or ( shape index, distance along shape )
            representation
        distance : float, optional
            distance to offset point from path (default is 0)

        Returns
        -------
        xy : (n,2) array
            x,y coordinates
        """
        # find index of shape context in self._shapes
        # if x is not a tuple, then convert to (shape, dist_along_shape)
        if not isinstance(x, tuple):
            x = self.path2shape(x)

        if isinstance(x[1], tuple):
            raise NotImplementedError()
            # does it happen that x[1] is a (nested) tuple (shape index, distance along shape)?
            # if so, we only support cases in which all shape indices in x[0] are equal
            # and what to do if forced_length is not None? Do we scale the nested distance?
            # What if the nested distance is another tuple?
            # if np.unique(x[0].ravel()).shape[0] != 1:
            #    raise ValueError("The use of `x` as a nested tuple (shape, (obj, distance)) is only supported for one shape.")
            # xy = self._shapes[x[0][0]].path2point(x[1])
        else:
            dist_along_shape = x[1]
            if not self.forced_length is None:
                dist_along_shape = (
                    dist_along_shape
                    * np.sum([k.pathlength for k in self._shapes])
                    / self.forced_length
                )
            xy = np.empty((dist_along_shape.size, 2)) * np.nan
            for k, p in enumerate(self._shapes):
                idx = x[0] == k
                xy[idx] = self._shapes[k].path2point(dist_along_shape[idx])

        return xy

    @property
    def pathlength(self):
        """Total path length of all shapes combined."""
        return np.sum(self.shapelengths)

    @property
    def shapelengths(self):
        """Path length of each shape in collection."""
        L = np.array([x.pathlength for x in self._shapes])

        if not self.forced_length is None:
            L = L * self.forced_length / np.sum(L)

        return L

    def path2shape(self, x):
        """Convert path to shape representation.

        Parameters
        ----------
        x : array
            distance along path

        Returns
        -------
        tuple
            representation of point on path as shape index and
            distance along shape.
        """
        return _path2edge(x, self.shapelengths, self.pathlength)

    def shape2path(self, x):
        """Convert shape to path representation.

        Parameters
        ----------
        x : tuple
            (shapeindex, distance along shape) representation of points on
            path

        Returns
        -------
        array
            distance along path
        """
        return _edge2path(x, self.shapelengths, self.pathlength)

    def samplepath(self, oversampling=20):
        """Regular sampling of points on path.

        Parameters
        ----------
        oversampling : int, optional
            Factor that determines how many more points should be sampled
            along path.

        Returns
        -------
        array
            x,y coordinates of sampled points
        """
        # xy = np.zeros( (0,2) )
        xy = []
        for p in self._shapes:
            # xy = np.concatenate( (xy,p.samplepath( oversampling=oversampling )) )
            xy.append(p.samplepath(oversampling=oversampling))
        return xy

    def bin(self, binsize, separate=True, recursive=True):
        """Divide path into bins.

        Parameters
        ----------
        binsize : float
            Desired bin size
        separate : bool
            treat graph edges separately
        recursive : bool
            pass `separate` argument to bin method of contained shapes

        Returns
        -------
        bins : 1d array
            bin edges
        nbins : int
            number of bins
        binsize : float
            actual bin size
        """
        if not separate:
            return _bin_path([self.pathlength], binsize)
        else:
            bins, nbins, binsize = self._shapes[0].bin(binsize, separate=recursive)
            bins, nbins, binsize = [bins], [nbins], [binsize]

            cum_lengths = np.cumsum(self.shapelengths)

            for p, l in zip(self._shapes[1:], cum_lengths[:-1]):
                a, b, c = p.bin(binsize, separate=recursive)
                bins.append(a[1:] + l)
                nbins.append(b)
                binsize.append(c)

            bins = np.concatenate(bins)
            nbins = np.concatenate(nbins)
            binsize = np.concatenate(binsize)

            return bins, nbins, binsize


# convenience functions
def ngon(size=1, n=3):
    """Generate a general n-sided polygon.

    Builds an n-sided polygon centered on (0,0) with the spoke lengths
    set by `size`.

    Parameters
    ----------
    size : scalar
    n : int

    Returns
    -------
    polygon

    """
    angle = np.arange(n) * 2 * np.pi / n
    vertices = size * np.vstack((np.cos(angle), np.sin(angle))).T
    return polygon(vertices=vertices)


def triangle(*args, **kwargs):
    """Generate an equal-sided triangle.

    Parameters
    ----------
    size : scalar

    Returns
    -------
    polygon

    """
    kwargs["n"] = 3
    return ngon(*args, **kwargs)


def pentagon(*args, **kwargs):
    """Generate an equal-sided pentagon.

    Parameters
    ----------
    size : scalar

    Returns
    -------
    polygon

    """
    kwargs["n"] = 5
    return ngon(*args, **kwargs)


def hexagon(*args, **kwargs):
    """Generate an equal-sided hexagon.

    Parameters
    ----------
    size : scalar

    Returns
    -------
    polygon

    """
    kwargs["n"] = 6
    return ngon(*args, **kwargs)


# helper functions
def _construct_spline(vertices, closed=False):
    s = sp.interpolate.splprep((vertices[:, 0], vertices[:, 1]), s=0.0, per=closed)
    # sder = sp.interpolate.splder( s[0] )
    return s[0:2]


def _sample_spline(vertices, oversampling=20, closed=False, openpath=False):

    oversampling = np.floor(oversampling)
    if oversampling == 1:
        return vertices.copy()

    nvertices = vertices.shape[0]
    npoints = nvertices + (oversampling - 1) * (nvertices - 1) - int(openpath)

    spline_param, spline_u = _construct_spline(vertices, closed)

    fcn = sp.interpolate.interp1d(
        np.arange(len(spline_u)), spline_u, kind="linear", axis=0, bounds_error=False
    )
    x = np.linspace(
        0, len(spline_u) - 1, num=npoints, endpoint=not (closed and openpath)
    )
    sampled_u = fcn(x)

    sampled_spline = np.empty((sampled_u.size, 2))
    sampled_spline[:, 0], sampled_spline[:, 1] = sp.interpolate.splev(
        sampled_u, spline_param
    )

    # xyder = np.empty( (u.size, 2) )
    # xyder[:,0],xyder[:,1] = sp.interpolate.splev( u, spline_param, der=1 )

    return ((sampled_spline, sampled_u), (spline_param, spline_u))


def _sample_polyline(vertices, oversampling=1, closed=False, openpath=False):
    oversampling = np.floor(oversampling)
    if oversampling == 1:
        return vertices.copy()

    nvertices = vertices.shape[0]

    npoints = nvertices + (oversampling - 1) * (nvertices - 1) - int(openpath)

    fcn = sp.interpolate.interp1d(
        np.arange(nvertices), vertices, kind="linear", axis=0, bounds_error=False
    )
    x = np.linspace(0, nvertices - 1, num=npoints, endpoint=not (closed and openpath))
    xy = fcn(x)

    return xy


def _inv_ellipeinc(a, b, n=100):

    if a > b:
        t_offset = 1.5 * np.pi
    else:
        t_offset = np.pi
        a, b = b, a

    eccentricity = 1 - (b / a) ** 2
    t = np.linspace(0, 1, num=n) * 2 * np.pi

    offset = a * (
        sp.special.ellipeinc(2 * np.pi, eccentricity)
        - sp.special.ellipeinc(t_offset, eccentricity)
    )
    s = np.mod(
        a * sp.special.ellipeinc(circ.wrap(t + t_offset), eccentricity) + offset,
        a * sp.special.ellipeinc(2 * np.pi, eccentricity),
    )
    s[-1] = a * sp.special.ellipeinc(2 * np.pi, eccentricity)

    fcn = sp.interpolate.interp1d(s, t, bounds_error=False)
    return fcn


def _check_graph(nodes, polylines, tol=0.001, correct=False):

    nodes = util.aspoints(nodes)

    if isinstance(polylines, polyline):
        polylines = [polylines]
    elif not all([isinstance(x, polyline) for x in polylines]):
        raise TypeError("Graph edges need to be polyline objects.")

    # check for duplicate nodes
    if nodes.shape[0] > 1:
        d = scipy.spatial.distance.pdist(nodes)
        d = scipy.spatial.distance.squareform(d)
        if np.min(d[np.tril_indices_from(d, k=-1)]) < tol:
            raise ValueError("Duplicate nodes found in graph.")

    npolylines = len(polylines)

    edges = np.zeros((npolylines, 2), dtype=np.int) - 1

    for index, p in enumerate(polylines):
        # find closest node for polyline start vertex
        d = scipy.spatial.distance.cdist(nodes, p.vertices[0:1])
        dmin = np.argmin(d)
        if d[dmin] <= tol:
            edges[index, 0] = dmin
            if correct and d[dmin] > 0:
                p.vertices[0] = nodes[dmin]

        # find closest node for polyline end vertex
        d = scipy.spatial.distance.cdist(nodes, p.vertices[-1:])
        dmin = np.argmin(d)
        if d[dmin] <= tol:
            edges[index, 1] = dmin
            if correct and d[dmin] > 0:
                p.vertices[-1] = nodes[dmin]

    return (nodes, polylines, edges)


def _path2edge(x, edgelengths, pathlength):
    n = len(edgelengths) + 1
    L = np.cumsum(np.concatenate(([0], edgelengths)))
    L[-1] = pathlength
    edge_index = np.concatenate((np.arange(n - 1), [n - 2]))

    valid = ~np.isnan(x)
    edge = np.full(x.shape, -1, dtype=np.int)
    dist_along_edge = np.full(x.shape, np.nan, dtype=np.float)

    edge[valid] = np.floor(
        sp.interpolate.interp1d(L, edge_index, kind="linear", axis=0)(x[valid])
    ).astype(np.int)
    dist_along_edge[valid] = x[valid] - L[edge[valid]]

    return (edge, dist_along_edge)


def _edge2path(x, edgelengths, pathlength):
    if not isinstance(x, tuple):
        raise ValueError(
            "Expecting a tuple (shapeindex, distance along shape) representation of points on path"
        )

    valid = np.logical_and(x[0] >= 0, ~np.isnan(x[1]))

    L = np.cumsum(np.concatenate(([0], edgelengths)))
    L[-1] = pathlength

    p = np.full(x[0].shape, np.nan, dtype=np.float)
    p[valid] = L[x[0][valid]] + x[1][valid]

    return p


def _bin_path(edge_lengths, dx):

    edge_lengths = np.asarray(edge_lengths)

    # fraction number of bins
    nbins = edge_lengths / dx
    # lower and upper integral number of bins
    nbins = np.vstack([np.floor(nbins), np.ceil(nbins)])

    # choose the number of bins that minimizes the difference between
    # the requested bin size and the actual bin size
    dx_prime = edge_lengths / nbins
    selection = np.argmin(np.abs(dx_prime - dx), axis=0)
    nbins = nbins[selection, np.arange(len(edge_lengths))]
    nbins = nbins.astype(np.int)

    # compute bin edges using interpolation
    cL = np.concatenate([[0], np.cumsum(edge_lengths)])
    cN = np.concatenate([[0], np.cumsum(nbins)])

    bins = scipy.interpolate.interp1d(cN, cL)(np.arange(cN[-1] + 1))

    return bins, nbins, edge_lengths / nbins
