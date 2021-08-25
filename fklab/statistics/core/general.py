"""
========================================================
General utilities (:mod:`fklab.statistics.core.general`)
========================================================

.. currentmodule:: fklab.statistics.core.general

General utilities.


"""
from fklab.version._core_version._version import __version__

__all__ = [
    "format_pvalue_stars",
    "format_pvalue_exact",
    "format_pvalue_relative",
    "format_pvalue",
    "monte_carlo_pvalue",
    "find_mode",
    "beta_reparameterize",
    "map_array",
]

import re

import numpy as np
import scipy as sp
import scipy.stats


def _check_significance_levels(alpha):
    """Check vector of significance levels

    Parameters
    ----------
    alpha : None or 1D array-like
        A list of significance levels. If None, the default
        `[0.001, 0.01, 0.05]` is used.

    Returns
    -------
    alpha: 1d array
        1D array with checked significance levels that is
        padded with infinity values on both ends.

    """

    if alpha is None:
        alpha = np.array([0.001, 0.01, 0.05])
    else:
        alpha = np.sort(np.array(alpha, dtype=np.double).ravel())

    alpha = np.clip(alpha, 0, 1)

    if len(alpha) == 0:
        raise ValueError("Invalid array of significance levels")

    alpha = np.pad(alpha, (1, 1), constant_values=(-np.inf, np.inf))

    return alpha


def format_pvalue_stars(p, alpha=None, significant="*", insignificant="n.s.", **kwargs):
    """Format p-value qualitatively as stars.

    Parameters
    ----------
    p : float scalar
    alpha : 1D array-like
        Significance levels
    significant : str
        Text to be used to indicate significance level
    insignificant : str
        Text to be used to indicate insignificant p-value

    Returns
    -------
    str : formatted p-value

    """

    p = np.clip(float(p), 0, 1)

    alpha = _check_significance_levels(alpha)

    idx = np.digitize(p, alpha)

    if idx >= len(alpha) - 1:
        return insignificant.format(alpha=alpha[-2], alpha_percent=100 * alpha[-2])
    else:
        return significant * (len(alpha) - idx - 1)


def format_pvalue_exact(p, precision=3, fmt="p={pvalue}", latex=False, **kwargs):
    """Format p-value as number.

    Parameters
    ----------
    p : float scalar
    precision : integer >= 0
        The number of digits to represent the p-value
    prefix : str
        Add prefix to formatted p-value
    latex : bool
        Use latex for scientific notation

    Returns
    -------
    str : formatted p-value

    """

    p = np.clip(float(p), 0, 1)

    if p != 0 and abs(p) < 10 ** (-precision):
        p = r"{:.{}e}".format(p, precision - 1)
        if latex:
            c = re.compile(
                r"(?P<base>[+-]?[0-9.]+)[eE](?P<sign>-?)[+0]*(?P<exp>[0-9]+)"
            )
            p = c.sub(r"$\g<base>{\\times}10^{\g<sign>\g<exp>}$", p)
    else:
        p = "{:.{}f}".format(p, precision)
        if latex:
            p = r"${}$".format(p)

    return fmt.format(pvalue=p)


def format_pvalue_relative(
    p, alpha=None, significant="p<{alpha}", insignificant="p≥{alpha}", **kwargs
):
    """Format p-value relative to significance level.

    Parameters
    ----------
    p : float scalar
    alpha : 1D array-like
        Significance levels
    insignificant : str or None
        Text to be used to indicate insignificant p-value.
    prefix : str
        Add prefix to formatted p-value

    Returns
    -------
    str : formatted p-value
    """

    p = np.clip(float(p), 0, 1)

    alpha = _check_significance_levels(alpha)

    idx = np.digitize(p, alpha)

    if idx >= len(alpha) - 1:
        return insignificant.format(alpha=alpha[-2], alpha_percent=100 * alpha[-2])
    else:
        return significant.format(alpha=alpha[idx], alpha_percent=100 * alpha[idx])


def format_pvalue(p, kind="exact", **kwargs):
    """String formating of p-value.

    Parameters
    ----------
    p : float, 0<=p<=1
    kind : 'stars', 'exact' or 'relative'
    **kwargs :
        Extra keyword arguments for the formatting function.

    Returns
    -------
    str : formatted p-value

    """

    if kind == "stars":
        return format_pvalue_stars(p, **kwargs)
    elif kind == "exact":
        return format_pvalue_exact(p, **kwargs)
    elif kind == "relative":
        return format_pvalue_relative(p, **kwargs)
    else:
        raise ValueError("Unsupported value for kind argument")


def monte_carlo_pvalue(simulated, test, tails="right", center=0, axis=0):
    """Compute Monte Carlo p-value.

    Computes (r+1)/(n+1), where n is the number of simulated samples and
    r is the number of simulated sample with a test statistic greater than or
    equal to the test statistic derived from the actual data.

    Parameters
    ----------
    simulated : 1d array
        Simulated test statistics
    test : scalar
        Test statistic derived from actual data

    Returns
    -------
    p : scalar
        Monte Carlo p-value

    """
    simulated = np.atleast_1d(simulated)
    test = np.atleast_1d(test)

    if test.size != int(np.prod(simulated.shape[1:])):
        raise ValueError("Incorrect size of test statistic array.")

    if test.ndim == simulated.ndim - 1:
        test = test[None, :]
    elif test.ndim != simulated.ndim or test.shape[0] != 1:
        raise ValueError("Incompatible shape of test statistic array.")

    if tails == "right":
        cmp_fcn = np.greater_equal
    elif tails == "left":
        cmp_fcn = np.less_equal
    elif tails == "both":
        cmp_fcn = lambda x, y: np.abs(x) >= np.abs(y)
    else:
        raise ValueError("Invalid tails.")

    if center is None:
        center = lambda x: np.nanmean(x, axis=0)

    if callable(center):
        center = center(simulated)
    else:
        center = np.asarray(center)

    if center.ndim > 0:
        if center.size != test.size:
            raise ValueError("Incorrect size of center array.")
        center = center.reshape(test.shape)

    p = (np.nansum(cmp_fcn(simulated - center, test - center), axis=axis) + 1) / (
        len(simulated) + 1
    )

    return p


def find_mode(x, axis=0, gridsize=201):
    """Find mode in observations.

    To determine the mode of a set of observations, a kernel
    density estimate (kde) of the distribution is computed and
    evaluated at a grid. The mode is then taken as the point at
    which the kde is maximal.

    Parameters
    ----------
    x : array
    axis : int
        axis along which to find the mode
    gridsize : int
        number of grid points to compute kernel density estimate

    Returns
    -------
    mode : array

    """
    shape = x.shape
    ndim = x.ndim

    # reshape
    x = np.moveaxis(x, axis, -1)
    new_shape = x.shape[:-1] if ndim > 1 else (1,)
    x = np.reshape(x, (x.size // shape[axis], shape[axis]))

    l, u = np.min(x, axis=1), np.max(x, axis=1)

    mode = np.full(x.shape[0], np.nan)

    for k, y in enumerate(x):
        grid = np.linspace(l[k], u[k], gridsize)
        density = sp.stats.gaussian_kde(y)(grid)
        mode[k] = grid[np.argmax(density)]

    mode = np.reshape(mode, new_shape)

    if ndim == 1:
        mode = float(mode)

    return mode


def beta_reparameterize(mode=0.5, concentration=4):
    """Reparameterization of beta distribution.

    Parameters
    ----------
    mode : float in range [0,1]
    concentration : float >= 2

    Returns
    -------
    alpha, beta : float

    """
    return (mode * (concentration - 2) + 1, (1 - mode) * (concentration - 2) + 1)


def map_array(index, *args, fcn="count"):
    """Split-apply-combine operation.

    Similar to scipy.stats.binned_statistic_dd with the following differences:
    1. binned_statistic performs binning, map_array requires binned data
       (to make function simpler and allow for flexibility in binning strategy)
    2. binned_statistic takes value arrays that have shape (n,) and computes
       statistics on each array separately; map_array takes value arrays that
       have shape (n,...) and passes all value arrays to the function as
       multiple arguments
    3. binned_statistic only supports custom function that return scalar;
       map_array supports functions that return array and supports both mapping
       and reduction operations
    4. binned_statistic returns nd array with result; map_array returns (n,...)
       array (for mapping) or (nbins,...) array (for reduction)

    Parameters
    ----------
    index : (n,...) array
        Array of group indices for data samples, which is generally the
        result of a prior grouping/binning operation.
    *args : (n,...) arrays
        Data arrays in which the first dimension is the same as the number
        of samples in the index array.
    fcn : callable
        Function that takes the split data arrays as arguments and returns
        an array. The returned array is either the result of a mapping
        operation on each sample so that the size of the first dimension
        is unchanged (but other dimensions may change) or the result of a
        reduction operation on the group of samples so that the size of the
        first dimension is 1 (or generally different from the number of input
        samples).

    Returns
    -------
    result : array
        In case of a mapping operation, *result* is a (n,...) array.
        In case of a reduction operation, *result* is a (ngroups,...) array.
    groups : array
        Indices of groups.

    """

    # check arguments
    index = np.atleast_1d(index)

    n = len(index)

    if len(args) == 0:
        args = [np.ones((n,))]
    else:
        args = [np.atleast_1d(arg) for arg in args]
        if not all([len(arg) == n for arg in args]):
            raise ValueError(
                "All data arrays should have first dimension of size {}".format(n)
            )

    # find unique indices
    unique_index, unique_inverse, unique_count = np.unique(
        index, return_inverse=True, return_counts=True, axis=0
    )

    # short cut for count reduction
    if fcn is None or fcn == len or fcn == "count":
        return unique_count, unique_index

    sorted_inverse = np.argsort(unique_inverse)
    idx = np.concatenate([[0], np.cumsum(unique_count)])

    # we try to auto detect mapping or reduction operation
    # if size of first dimension for all groups equals the
    # number of input samples, then it is a mapping operation
    # otherwise it is a reduction operation
    reduce = False

    result = []
    for k, (start, stop) in enumerate(zip(idx[:-1], idx[1:])):
        data = np.atleast_1d(fcn(*[v[sorted_inverse[start:stop]] for v in args]))
        reduce = reduce or len(data) != (stop - start)
        result.append(data)

    if reduce:
        if all([r.shape[0] == 1 for r in result]):
            result = np.concatenate(result)
        else:
            result = np.stack(result)
    else:
        result = np.concatenate(result)
        result = result[np.argsort(sorted_inverse)]

    return result, unique_index
