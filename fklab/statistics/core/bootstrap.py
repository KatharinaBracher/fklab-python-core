"""
======================================================
Bootstrap (:mod:`fklab.statistics.core.bootstrap`)
======================================================

.. currentmodule:: fklab.statistics.core.bootstrap

Bootstrapping utilities.

.. autosummary::
    :toctree: generated/

    ci

"""
import numpy as np

from fklab.version._core_version import __version__


def ci(
    data, nsamples=10000, statistic=None, alpha=0.05, alternative="two-sided", axis=0
):
    """Bootstrap estimate of 100.0*(1-alpha) confidence interval for statistic.

    Parameters
    ----------
    data : array or sequence of arrays
    nsamples : int, optional
        number of bootstrap samples
    statistic : function( data ) -> scalar or array, optional
    alpha : float, optional
    alternative: "two-sided", "greater" or "less"
    axis : int, optional
        axis of data samples

    Returns
    -------
    low,high : confidence interval

    """

    if not isinstance(data, (list, tuple)):
        data = (data,)

    if statistic is None:
        statistic = lambda x: np.average(x, axis=axis)

    if not alternative in ["two-sided", "greater", "less"]:
        raise ValueError("Invalid alternative argument")

    bootindexes = bootstrap_indexes([x.shape[axis] for x in data], nsamples)
    stat = np.array(
        [
            statistic(*(x.take(idx, axis=axis) for x, idx in zip(data, indexes)))
            for indexes in bootindexes
        ]
    )
    # stat.sort(axis=0)

    if alternative == "two-sided":
        lo, hi = np.nanpercentile(
            stat, [100.0 * alpha / 2.0, 100.0 * (1 - alpha / 2.0)], axis=0
        )
    elif alternative == "greater":
        lo, hi = -np.inf, np.nanpercentile(stat, 100.0 * (1 - alpha), axis=0)
    else:  # 'less'
        lo, hi = np.nanpercentile(stat, 100.0 * alpha, axis=0), np.inf

    return lo, hi


def bootstrap_indexes(n, nsamples=10000):
    """Generator for bootstrap indices.
    """
    if not isinstance(n, (list, tuple)):
        n = (n,)

    for _ in range(nsamples):
        yield (np.random.randint(N, size=(N,)) for N in n)
