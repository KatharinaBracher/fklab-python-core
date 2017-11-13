"""
========================================================
General utilities (:mod:`fklab.statistics.core.general`)
========================================================

.. currentmodule:: fklab.statistics.core.general

General utilities.

.. autosummary::
    :toctree: generated/
    
    monte_carlo_pvalue

"""

__all__ = ['monte_carlo_pvalue']

import numpy as np

def monte_carlo_pvalue(simulated, test, tails='right', center=0, axis=0):
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
    
    if tails == 'right':
        cmp_fcn = np.greater_equal
    elif tails == 'left':
        cmp_fcn = np.less_equal
    elif tails == 'both':
        cmp_fcn = lambda x, y: np.abs(x)>=np.abs(y)
    else:
        raise ValueError("Invalid tails.")
    
    if center is None:
        center is np.nanmean
    
    if callable(center):
        center = center(simulated)
    
    p = (np.nansum(cmp_fcn(simulated-center,test-center), axis=axis)+1)/(len(simulated)+1)
    
    return p

