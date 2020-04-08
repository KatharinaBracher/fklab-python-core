"""
======================================================================
Kernel density smoothing (:mod:`fklab.signals.smooth.kernelsmoothing`)
======================================================================

.. currentmodule:: fklab.signals.smooth.kernelsmoothing

Classes and function for data smoothing.

"""
import numpy as np
import scipy.signal

from fklab.version._core_version._version import __version__

__all__ = [
    "GaussianKernel",
    "EpanechnikovKernel",
    "UniformKernel",
    "TriangularKernel",
    "VonMisesKernel",
    "MixedKernel",
    "NoKernel",
    "create_smoother",
    "Smoother",
    "smooth1d",
    "smooth2d",
]


def create_smoother(
    ndim,
    kernel="gaussian",
    bandwidth=0,
    delta=None,
    unbiased=True,
    nansaszero=True,
    axes=None,
):
    """Create a Smoother instance.

    Parameters
    ----------
    ndim : int
        Number of data dimensions.

    kernel : "gaussian", "epanechnikov", "uniform" or "triangular"
        Type of smoothing kernel. The same kernel will be used for all dimensions.

    bandwidth : float or [float,...]
        Bandwidth of the smoothing kernel. If a float, then the same bandwidth is used
        for all dimensions. If a sequence of floats, then the length of the sequence
        should be *ndim* or len(axes) if the *axes* argument is set.

    delta : None, float or [float,...]
        Spacing between data points along each array dimension. If None, spacing is
        fixed to 1. If a float, then the same spacing is used for all dimensions.
        If a sequence of floats, then the length of the sequence should be *ndim* or
        len(axes) if the *axes* argument is set.

    unbiased : bool
        Only use available data for smoothing. With unbiased smoothing there is no
        zero-padding at the edges of an array. In combination with *nansaszero*, NaNs
        will be removed for smoothing purposes.

    nansaszero : bool
        Convert NaNs to zero before smoothing. In combination with *unbiased*, NaNs
        will be considered invalid data and will not be used in smoothing even after
        after conversion to zeros.

    axes : None or [int,...]
        Selected array dimensions that should be smoothed. A *NoKernel* kernel
        (i.e. no smoothing) is used for all other dimensions. A dimension can
        only be listed once and the order in which dimensions are listed should
        match the order in which bandwidths and deltas are provided.

    Returns
    -------
    smoother(data, delta=None)
        Function that smooths a *ndim* array with the given parameters.
        Optionally, the delta keyword argument can be set to override the
        delta parameter supplied to the *create* function.

    """
    # CHECK ARGUMENTS

    ndim = int(ndim)
    if ndim < 0:
        raise ValueError("Number of dimensions ndim expected to be >=0.")
    if ndim == 0:
        return lambda data: data

    if not kernel in ("gaussian", "epanechnikov", "uniform", "triangular"):
        raise ValueError(
            "Valid kernels are: gaussian, epanechnikov, uniform and triangular"
        )

    kernel = _kernel_map[kernel]

    if axes is None:
        axes = np.arange(ndim, dtype=np.int)
    else:
        axes = np.asarray(axes, dtype=np.int).ravel()
        if len(axes) == 0:
            axes = np.arange(ndim, dtype=np.int)
        elif len(axes) > ndim:
            raise ValueError(
                "The size of axes cannot be higher than ndim ({}).".format(ndim)
            )

    if len(np.unique(axes)) != len(axes):
        raise ValueError("Duplicate entries in axes are not allowed.")

    if any([x < 0 or x > ndim - 1 for x in axes]):
        raise ValueError("Entries in axes should be in range [0,{}]".format(ndim - 1))

    axes_ndim = len(axes)

    bandwidth = np.atleast_1d(bandwidth).ravel()
    if len(bandwidth) == 1:
        bandwidth = np.ones(axes_ndim) * bandwidth
    elif len(bandwidth) != axes_ndim:
        raise ValueError(
            "Expecting scalar or ({},) array-like bandwidth.".format(axes_ndim)
        )

    if any([b < 0 for b in bandwidth]):
        raise ValueError("Bandwidth cannot be negative.")

    if delta is None:
        delta = np.ones(axes_ndim)
    else:
        delta = np.atleast_1d(delta).ravel()
        if len(delta) == 1:
            delta = np.ones(axes_ndim) * delta
        elif len(delta) != axes_ndim:
            raise ValueError(
                "Expecting scalar or ({},) array-like delta.".format(axes_ndim)
            )

        if any([d <= 0 for d in delta]):
            raise ValueError("Delta cannot be zero or negative.")

    full_delta = np.ones(ndim)
    full_delta[axes] = delta

    # END CHECK ARGUMENTS

    # special case of no smoothing
    if np.allclose(bandwidth, 0.0):
        return lambda data: data

    # construct kernel
    K = [NoKernel()] * ndim
    for ax, bw in zip(axes, bandwidth):
        if not np.isclose(bw, 0.0):
            K[ax] = kernel(bandwidth=bw)

    K = MixedKernel(*K)

    # construct smoother object
    _smoother = Smoother(kernel=K, unbiased=unbiased, nansaszero=nansaszero)

    def smoother(data, delta=None):
        if not delta is None:
            actual_delta = full_delta.copy()
            actual_delta[axes] = delta
        else:
            actual_delta = full_delta

        return _smoother(data, actual_delta)

    return smoother


class KernelBase(object):
    def __init__(self):
        pass

    def sample(self, dx):
        raise NotImplementedError


class LinearKernelBase(KernelBase):
    """Define base class for linear kernel functions.

    Parameters
    ----------
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    """

    def __init__(
        self,
        bandwidth=None,
        covariance=None,
        correlation=None,
        kerneltype="symmetrical",
        **kwargs
    ):
        KernelBase.__init__(self, **kwargs)

        if covariance is not None and (
            bandwidth is not None or correlation is not None
        ):
            raise TypeError

        if covariance is not None:
            self.covariance = covariance
        else:
            self._covariance = self._bw2cov(bandwidth, correlation)

        self._support = 1

        self.kerneltype = kerneltype

    def _cov2bw(self, Z):
        if Z is None:
            return np.array([1]), np.matrix(1)

        Z = np.asmatrix(Z, dtype=np.float64)

        bw = np.diag(Z) ** 0.5

        d = np.linalg.inv(np.diag(bw))
        p = np.asmatrix(d * Z * d, dtype=np.float64)

        return bw, p

    def _bw2cov(self, bw, p):
        if bw is None and p is None:
            return np.matrix(1)

        if bw is None:
            if p.shape[0] != p.shape[1] or np.any(np.abs(p.ravel()) >= 1):
                raise TypeError
            bw = np.ones(len(p))

        if p is None:
            bw = np.asarray(bw, dtype=np.float64)
            if bw.ndim == 0:
                bw = bw.reshape((1,))
            if bw.ndim > 1 or np.any(bw <= 0):
                raise TypeError
            p = np.identity(len(bw))

        p = np.asmatrix(p, dtype=np.float64)

        d = np.diag(bw)
        Z = d * p * d

        return Z

    @property
    def kerneltype(self):
        """Symmetrical or multiplicative kernel."""
        return self._kerneltype

    @kerneltype.setter
    def kerneltype(self, value):
        if not value in ("symmetrical", "multiplicative"):
            raise TypeError
        self._kerneltype = value

    @property
    def support(self):
        """Finite support."""
        return self._support

    @support.setter
    def support(self, value):
        value = float(value)
        if value <= 0:
            raise TypeError
        self._support = value

    @property
    def ndim(self):
        """Dimensionality of kernel."""
        return len(self._covariance)

    @property
    def bandwidth(self):
        """Kernel bandwidth."""
        bw, p = self._cov2bw(self.covariance)
        return bw

    @bandwidth.setter
    def bandwidth(self, value):
        # convert to array
        value = np.array(value, dtype=np.float64)
        # check if 1D vector
        if value.ndim == 0:
            value = value.reshape((1,))
        elif value.ndim > 1:
            raise TypeError
        # check if same size as covariance matrix
        if len(value) == 1:
            value = value * np.ones(len(self._covariance))
        elif len(value) != len(self._covariance):
            raise TypeError
        if np.any(value <= 0):
            raise TypeError

        self._covariance = self._bw2cov(value, self.correlation)

    @property
    def correlation(self):
        """Correlation matrix."""
        bw, p = self._cov2bw(self.covariance)
        return p

    @correlation.setter
    def correlation(self, value):
        # convert to matrix
        value = np.asmatrix(value, dtype=np.float64)
        # check if square
        if value.shape[0] != value.shape[1]:
            raise TypeError
        # check if symmetric
        if not np.all(value == value.T):
            raise TypeError
        # check if equal size as covariance matrix
        if len(value) != len(self._covariance):
            raise TypeError
        # check if in range <-1,1>
        if np.any(value < -1) or np.any(value > 1) or np.any(np.diag(value) != 1):
            raise TypeError

        self._covariance = self._bw2cov(self.bandwidth, value)

    @property
    def covariance(self):
        """Covariance matrix."""
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        # convert to matrix
        value = np.asmatrix(value, dtype=np.float64)
        # check if square matrix
        if value.shape[0] != value.shape[1]:
            raise TypeError
        # check if symmetric
        if not np.all(value == value.T):
            raise TypeError
        # check if positive semi definite
        if not np.all(np.linalg.eigvalsh(value) > -1e-8):
            raise TypeError

        self._covariance = value

    def _kernel_function(self, u):
        raise NotImplementedError

    def _multiplicative_kernel(self, dx):
        # for each dimension
        # define distance vector
        # compute the 1D kernel
        # and multiply with 1D kernels in other dimensions
        bw, p = self._cov2bw(self._covariance)
        k = np.array(1)
        for idx, b in enumerate(bw):
            u, npoints = self._compute_distance_array(
                np.array([b ** 2]), np.array([dx[idx]])
            )
            p = self._kernel_function(u)
            k = k[..., np.newaxis] * p.reshape([1] * idx + [len(p)])

        return k

    def _symmetrical_kernel(self, dx):
        # define distance matrix
        # compute 1/det(H) * K( H^-1 * x )
        u, npoints = self._compute_distance_array(self._covariance, dx)
        p = self._kernel_function(u)
        p = p.reshape(tuple(npoints))
        return p

    def _compute_distance_array(self, covariance, dx):

        bw, p = self._cov2bw(covariance)
        npoints = np.ceil((self._support * bw) / dx)
        x = [np.arange(-b, b + 1) * a for a, b in zip(dx, npoints)]

        if len(x) == 1:
            grid = x[0]
            u = (grid ** 2) / np.asarray(covariance)
        else:
            grid = np.meshgrid(*x, indexing="ij")
            grid = [g.ravel() for g in grid]
            grid = np.vstack(grid)
            u = np.sum(np.asarray(np.linalg.inv(covariance) * grid) * grid, axis=0)

        return u, npoints.astype(np.int) * 2 + 1

    def __call__(self, dx):
        """Evaluate kernel function.

        Parameters
        ----------
        dx : number or 1D array

        Returns
        -------
        values : ndarray

        """
        dx = np.array(dx, dtype=np.float64)
        if dx.ndim == 0:
            dx = dx.reshape((1,))
        elif dx.ndim > 1:
            raise TypeError

        if len(dx) == 1:
            dx = dx * np.ones(len(self._covariance))
        elif len(dx) != len(self._covariance):
            raise TypeError

        if np.any(dx <= 0):
            raise TypeError

        if self.kerneltype == "multiplicative":
            return self._multiplicative_kernel(dx)
        elif self.kerneltype == "symmetrical":
            return self._symmetrical_kernel(dx)


class GaussianKernel(LinearKernelBase):
    """Define Gaussian kernel function.

    GaussianKernel(support=4, bandwidth=None, covariance=None, correlation=None, kerneltype='symmetrical')

    Parameters
    ----------
    support : scalar, optional
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    """

    def __init__(self, support=4, **kwargs):
        LinearKernelBase.__init__(self, **kwargs)
        self.support = support

    def _kernel_function(self, u):
        return np.exp(-0.5 * (u)) / np.sqrt(2 * np.pi)


class EpanechnikovKernel(LinearKernelBase):
    """Define Epanechnikov kernel function.

    EpanechnikovKernel(bandwidth=None, covariance=None, correlation=None, kerneltype='symmetrical')

    Parameters
    ----------
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    """

    def _kernel_function(self, u):
        val = np.zeros(u.shape)
        # u = u**2
        val[u < 1] = 0.75 * (1 - u[u < 1])
        return val


class UniformKernel(LinearKernelBase):
    """Define Uniform kernel function.

    UniformKernel(bandwidth=None, covariance=None, correlation=None, kerneltype='symmetrical')

    Parameters
    ----------
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    """

    def _kernel_function(self, u):
        val = np.zeros(u.shape)
        val[u < 1] = 1.0
        return val


class TriangularKernel(LinearKernelBase):
    """Define Triangular kernel function.

    TriangularKernel(bandwidth=None, covariance=None, correlation=None, kerneltype='symmetrical')

    Parameters
    ----------
    bandwidth : 1D array or list, optional
    covariance : 2D array, optional
    correlation : 2D array, optional
    kerneltype : {'symmetrical', 'multiplicative'}

    """

    def _kernel_function(self, u):
        val = np.zeros(u.shape)
        u = np.sqrt(u)
        val[u < 1] = 1 - u[u < 1]
        return val


class VonMisesKernel(KernelBase):
    pass


class MixedKernel(KernelBase):
    """Define Mixed kernel function.

    Parameters
    ----------
    *args : kernel objects

    """

    def __init__(self, *args):
        if not np.all([(x is None) or isinstance(x, KernelBase) for x in args]):
            raise TypeError

        self._kernels = list(args)

    @property
    def ndim(self):
        """Dimensionality of kernel."""
        d = 0
        for k in self._kernels:
            if k is None:
                d = d + 1
            else:
                d = d + k.ndim
        return d

    def __call__(self, dx):
        dx = np.array(dx, dtype=np.float64)
        if dx.ndim == 0:
            dx = dx.reshape((1,))
        elif dx.ndim > 1:
            raise TypeError

        if len(dx) == 1:
            dx = dx * np.ones(self.ndim)
        elif len(dx) != self.ndim:
            raise TypeError

        if np.any(dx <= 0):
            raise TypeError

        k = np.array(1)

        for idx, x in enumerate(self._kernels):

            k = k[..., None]

            if x is None:
                pass
            else:
                val = x(dx[idx])
                k = k * val.reshape([1] * (k.ndim - 1) + list(val.shape))

        return k


class NoKernel(KernelBase):
    """Define Non-smoothing dummy kernel.

    Parameters
    ----------
    ndim : integer

    """

    def __init__(self, ndim=1):
        self.ndim = ndim

    def __call__(self, dx=1):
        return np.array([1]).reshape(tuple([1] * self._ndim))

    @property
    def ndim(self):
        """Dimensionality of kernel."""
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        self._ndim = int(value)


class Smoother(object):
    """Define Smoothing class.

    Parameters
    ----------
    kernel : kernel object
    unbiased : bool
        Only take into account available data at edges
    nansaszero : bool
        Treat NaN in data as zeros
    normalize : bool or {'sum','max','none'}
        Method of kernel normalization.

    """

    def __init__(self, kernel=None, unbiased=False, nansaszero=False, normalize=True):
        self.kernel = kernel
        self.unbiased = unbiased
        self.nansaszero = nansaszero
        self.normalize = normalize

    @property
    def kernel(self):
        """Smoothing kernel."""
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        if not isinstance(value, KernelBase):
            raise TypeError
        self._kernel = value

    @property
    def unbiased(self):
        """Whether smoothing is unbiased."""
        return self._unbiased

    @unbiased.setter
    def unbiased(self, value):
        self._unbiased = bool(value)

    @property
    def nansaszero(self):
        """Whether NaNs are reated as zeros."""
        return self._nansaszero

    @nansaszero.setter
    def nansaszero(self, value):
        self._nansaszero = bool(value)

    @property
    def normalize(self):
        """Kernel normalization method."""
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        if isinstance(value, str):
            if not value in ("sum", "max", "none"):
                raise ValueError("Only 'sum', 'max' and 'none' are supported.")
            else:
                self._normalize = value
        elif value is None:
            self._normalize = "none"
        elif bool(value) == True:
            self._normalize = "sum"
        elif bool(value) == False:
            self._normalize = "none"
        else:
            raise ValueError("Invalid value.")

    def __call__(self, data, delta=1, method="auto"):
        """Smooth data.

        Parameters
        ----------
        data : ndarray
            The dimensionality of the data should match the dimensionality
            of the kernel.
        delta : scalar or sequence
            sampling interval of the data
        method : str {'auto', 'fft', 'direct'}
            numpy.convolve method

        Returns
        -------
        ndarray

        """
        data = np.array(data)
        if len(data) == 0:
            return data

        k = self._kernel(delta).copy()

        if self._normalize == "sum":
            k = k / np.nansum(k)
        elif self._normalize == "max":
            k = k / np.nanmax(k)

        if self._nansaszero:
            nan_data = np.isnan(data)
            data = data.copy()
            data[nan_data] = 0

            nan_kernel = np.isnan(k)
            k[nan_kernel] = 0
        else:
            nan_data = None

        data = scipy.signal.convolve(data, k, "same", method=method)

        if self._unbiased:
            n = np.ones(data.shape) / np.nansum(k)
            if nan_data is not None:
                n[nan_data] = 0
            n = scipy.signal.convolve(n, k, "same", method=method)
            if nan_data is not None:
                n[nan_data] = np.nan
            data = data / n

        return data


_kernel_map = {
    "none": NoKernel,
    "gaussian": GaussianKernel,
    "epanechnikov": EpanechnikovKernel,
    "uniform": UniformKernel,
    "triangular": TriangularKernel,
    "vonmises": VonMisesKernel,
}


def smooth1d(data, axis=-1, kernel="gaussian", bandwidth=1.0, delta=1.0, **kwargs):
    """Smooth array of 1D signals.

    Parameters
    ----------
    data : array
    axis : scalar, optional
        axis of array along which to perform smoothing
    kernel : {'gaussian', 'epanechnikov', 'uniform', 'triangular'}, optional
    bandwidth : scalar, optional
        bandwidth of kernel
    delta : scalar, optional
        sample period of data
    unbiased, nansaszero, normalized : see `Smoother`

    Returns
    -------
    signal : array
        smoothed data array

    """
    data = np.asarray(data)

    K = [NoKernel()] * data.ndim
    K[axis] = _kernel_map[kernel.lower()](bandwidth=bandwidth)

    K = MixedKernel(*K)

    smoother = Smoother(kernel=K, **kwargs)

    data = smoother(data, delta=delta)

    return data


def smooth2d(
    data, axes=[-1, -2], kernel="gaussian", bandwidth=1.0, delta=1.0, **kwargs
):
    """Smooth array of 2D arrays.

    Parameters
    ----------
    data : array
    axes : 2-element sequence
        the two axes of the data array along which to perform smoothing
    kernel : str or 2-element sequence, optional
        kernel type (one of 'gaussian', 'epanechnikov', 'uniform',
        'triangular') for each of the two dimensions
    bandwidth : scalar or 2-element sequence, optional
        bandwidths of kernel
    delta : scalar or 2-element sequence optional
        sample period of data along the two dimensions
    unbiased, nansaszero, normalized : see `Smoother`

    Returns
    -------
    signal : array
        smoothed data array

    """
    data = np.asarray(data)

    bandwidth = np.array(bandwidth).ravel()

    if data.ndim < 2:
        raise ValueError("Require at least 2 dimensions.")

    if len(axes) != 2 or axes[0] == axes[-1]:
        raise ValueError("Invalid axes")

    if not isinstance(kernel, (list, tuple)):
        kernel = [kernel]

    if len(kernel) < 1 or len(kernel) > 2:
        raise ValueError("Specify at least and at most 2 kernels for the 2 dimensions.")

    for k in kernel:
        if k not in list(_kernel_map.keys()):
            raise ValueError("Unknown kernel")

    K = [NoKernel()] * data.ndim
    K[axes[0]] = _kernel_map[kernel[0].lower()](bandwidth=bandwidth[0])
    K[axes[1]] = _kernel_map[kernel[-1].lower()](bandwidth=bandwidth[-1])

    K = MixedKernel(*K)

    smoother = Smoother(kernel=K, **kwargs)

    data = smoother(data, delta=delta)

    return data
