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


def monte_carlo_pvalue(simulated, test):
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
    
    p = (np.nansum(simulated>=test)+1)/(len(simulated)+1)
    
    return p

