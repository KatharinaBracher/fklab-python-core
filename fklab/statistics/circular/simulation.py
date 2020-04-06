"""
======================================================================
Circular data simulation (:mod:`fklab.statistics.circular.simulation`)
======================================================================

.. currentmodule:: fklab.statistics.circular.simulation

Tool to create a dataset with circular dependent variable

.. autosummary::
    :toctree: generated/

    generate_linear_circular_dataset
"""
import numpy as np

import fklab.statistics.circular as circ

__all__ = ["generate_linear_circular_dataset"]


def generate_linear_circular_dataset(
    nsamples=100, xmin=0, xmax=1, intercept=0, slope=None, period=None, noise=1.0
):
    """Generate data with circular dependent variable.

    Parameters
    ----------
    nsamples : int
        Number of samples in data set.
    xmin, xmax : scalar
        Defines range from which values of x are uniformy drawn.
    intercept : scalar
    slope : scalar
    period : scalar
        Either slope or period should be specified, but not both.
        Period is defined as 2*pi/slope and defines the range in x
        over which the angle completes a full cycle.
    noise : scalar
        Kappa parameter of the Von Mises distributed noise that is
        added to the data. Lower values of kappa introduce more noise.

    Returns
    -------
    x, theta : 1d array

    """

    if slope is None and period is None:
        slope = 3 * np.pi / (xmax - xmin)
    elif slope is None:
        slope = 2 * np.pi / period

    x = np.random.uniform(xmin, xmax, nsamples)
    theta = circ.wrap(intercept + slope * x + np.random.vonmises(0, noise, nsamples))

    return x, theta
