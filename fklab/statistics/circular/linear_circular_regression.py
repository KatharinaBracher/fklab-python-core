"""
========================================================================================
Circular Linear regression (:mod:`fklab.statistics.circular.linear_circular_regression`)
========================================================================================

.. currentmodule:: fklab.statistics.circular.linear_circular_regression

regression tool

"""
import numba
import numpy as np
import scipy.optimize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y

import fklab.statistics.circular as circ
import fklab.statistics.core

__all__ = ["CircularLinearRegression", "linear_circular_regression"]


class CircularLinearRegression(BaseEstimator):
    """Circular Linear Regression class used in combination with scikit-learn tools.

    Parameters
    ----------
    min_period : scalar
        Smallest range in x over which the fitted line is allowed to wrap.
        If not provided, it will be set to the range of x values divided
        by the square root of the number of x values.
    regularization : float
        Weight (lambda) of the LASSO regularization. Only the slope coefficient
        is included in the regularizer.

    ..note:: This class follow the scikit-learn API to be able to use it with cross-validation facilities. To do the actual regression, it is preferable to use the linear_circular_regression method.

    """

    def __init__(self, min_period=None, regularization=0.0):
        self.min_period = min_period
        self.regularization = regularization

    def fit(self, X, y):
        """Follow the reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples,1)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)

        self.coef_, self.r_, *_ = linear_circular_regression(
            X,
            y,
            min_period=self.min_period,
            ci="none",
            regularization=self.regularization,
        )

        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def score(self, X, y):
        y_pred = self.predict(X)

        res = circ.diff(y, y_pred, directed=True)

        v1 = np.sum(circ.diff(res, circ.mean(res)[0]) ** 2)
        v2 = np.sum(circ.diff(y, circ.mean(y)[0]) ** 2)
        r2 = 1 - v1 / v2

        return r2

    def predict(self, X):
        """Predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, 1)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)

        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        yhat = circ.wrap(self.coef_[0] + self.coef_[1] * X)
        return yhat.ravel()


# define objective function
# to speed up the computation, we use numba
@numba.jit(nopython=True, nogil=True)
def obj_fcn(coef, x, theta, regularization=0.0):
    # circ.diff is not compatible with numba, thus below we
    # copy the essentials of circ.wrap and circ.diff implementations
    # the assumption is that theta is in the range [0, 2pi]
    yhat = coef[0] + coef[1] * x
    yhat = np.mod(np.mod(yhat, 2 * np.pi) + 2 * np.pi, 2 * np.pi)
    loss = np.mean((np.pi - np.abs(np.pi - np.abs(theta - yhat))) ** 2)
    # apply LASSO regularization (only slope parameter):
    loss = loss + regularization * np.abs(coef[1])
    return loss


def linear_circular_regression(
    x,
    theta,
    min_period=None,
    ci="none",
    alpha=0.05,
    nsamples=200,
    regularization=0.0,
    strategy="best1bin",
):
    """Compute the Linear-circular regression.

    Computes the regression of circular variable *theta* on linear regressor
    *x*. The model is: theta = A + Bx modulus 2pi. The function will calculate
    the coefficients A and B by minimizing the sum of squared circular
    distances.

    Parameters
    ----------
    x : 1d array
        Linear regressor.
    theta : 1d array
        Circular dependent variable.
    min_period : scalar
        Smallest range in x over which the fitted line is allowed to wrap
        (i.e. span a full 2*pi cycle). If not provided, it will be set to
        the range of x values divided by the square root of the number of
        x values.
    ci : 'none' or 'theory' or 'bootstrap'
        Compute confidence interval for offset and slope. Bootstrapping does
        not compute the confidence interval for the intercept.
    alpha : float
        Confidence level.
    nsamples : int
        Number of bootstrap samples.
    regularization : float
        Weight (lambda) of the LASSO regularization. Only the slope coefficient
        is included in the regularizer. Regularization penalizes for large
        values of the slope parameter (i.e. a model that would wrap around many
        times over the range of x).

    strategy: str
        see strategy parameter in scipy.optimize.differential_evolution doc

    Returns
    -------
    offset,slope : scalars
    r : scalar
        Correlation coefficient.
    fcn : function
        Function to compute the best fitting line. It will take an array
        of x values.
    res : array
        Residuals.

    """
    # check arguments

    x = np.asarray(x).ravel()
    theta = np.asarray(theta).ravel()
    theta = circ.wrap(theta)

    if x.ndim != 1 or x.shape != theta.shape:
        raise ValueError("x and theta need to be 1d arrays of equal size")

    # exclude nans
    valid = ~np.logical_or(np.isnan(x), np.isnan(theta))
    x = x[valid]
    theta = theta[valid]

    npoints = len(x)

    if min_period is None:
        # heuristic to limit slope values
        min_period = (np.max(x) - np.min(x)) / np.sqrt(npoints)
    else:
        min_period = float(min_period)

    # perform minimization
    bounds = [(0, 2 * np.pi), (-2 * np.pi / min_period, 2 * np.pi / min_period)]

    result = scipy.optimize.differential_evolution(
        obj_fcn, bounds, args=(x, theta, regularization), strategy=strategy
    )

    if not result.success:
        raise ValueError("Minimization failed with message: {}".format(result.message))

    offset, slope = result.x

    # calculate residuals
    res = circ.diff(theta, (offset + slope * x), directed=True)

    if ci == "theory":
        # confidence intervals, assuming normal distribution of errors
        ssx = np.sum((x - np.mean(x)) ** 2)
        mse = np.sum(res ** 2) / (npoints - 2)

        t = np.array([-1, 1]) * scipy.stats.t.ppf(1 - alpha / 2, npoints - 2)
        ci_slope = slope + t * np.sqrt(mse / ssx)
        ci_offset = offset + t * np.sqrt(
            mse * ((1 / npoints) + (np.mean(x) ** 2) / ssx)
        )

        # pval = scipy.stats.t.cdf(slope/np.sqrt(mse/ssx), npoints-2)

    elif ci == "bootstrap":

        def statistic(data):
            result = scipy.optimize.differential_evolution(
                obj_fcn,
                bounds,
                args=(data[:, 0], data[:, 1], regularization),
                strategy=strategy,
            )
            return result.x

        ci_low, ci_high = fklab.statistics.core.ci(
            np.column_stack([x, theta]), statistic=statistic, nsamples=nsamples, axis=0
        )

        # the fklab.statistics.core.ci function uses np.nanpercentile
        # to compute the confidence interval, but that does not work for
        # circular data like the intercept

        # ci_offset = (ci_low[0], ci_high[0])
        ci_offset = None
        ci_slope = (ci_low[1], ci_high[1])

    else:
        ci_offset, ci_slope = None, None

    # compute correlation coefficient
    v1 = np.sum(circ.diff(res, circ.mean(res)[0]) ** 2)
    v2 = np.sum(circ.diff(theta, circ.mean(theta)[0]) ** 2)
    r = 1 - v1 / v2

    fcn = lambda x: circ.wrap(offset + slope * x)

    residuals = np.full(len(valid), np.nan)
    residuals[valid] = res

    return (offset, slope), r, fcn, residuals, (ci_offset, ci_slope)
