"""
==============================================================
Distance metrics (:mod:`fklab.statistics.distances.distances`)
==============================================================

.. currentmodule:: fklab.statistics.distances.distances

Functions to compute distance metrics.

"""
import numpy as np

__all__ = [
    "bhattacharyya_coefficient",
    "bhattacharyya_distance",
    "hellinger_distance",
    "prob_weighted_distance",
]


def bhattacharyya_coefficient(p1, p2, **kwargs):
    """Compute Bhattacharyya coefficient for two probability distributions.

    The Bhattacharyya coefficient is computed by summing the square root of
    the product of `p1` and `p2`.

    See https://en.wikipedia.org/wiki/Bhattacharyya_distance

    Parameters
    ----------
    p1, p2 : arrays
        Probability distributions
    axis : None, int or tuple of ints
        The array dimensions along which to compute the coefficient.
    keepdims : bool
        The `axis` dimensions are left in the result and not squeezed out.
    **kwargs :
        Optional arguments for `np.sum`

    Returns
    -------
    array

    Examples
    --------
    >>> bhattacharyya_coefficient(np.array([0.1, 0.3, 0.6]), np.array([0.2, 0.5, 0.3]))
    0.9529837595699797

    >>> bhattacharyya_coefficient(np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]]),
    ...                           np.array([[0.9, 0.0, 0.1], [0.2, 0.5, 0.3]]),
    ...                           axis=1)
    array([0.54494897, 0.86359859])

    """
    return np.sum(np.sqrt(p1 * p2), **kwargs)


def bhattacharyya_distance(p1, p2, **kwargs):
    """Compute Bhattacharyya distance between two probability distributions.

    The Bhattacharyya distance is computed as the negative natural
    logarithm of the Bhattacharyya coefficient.

    See https://en.wikipedia.org/wiki/Bhattacharyya_distance

    Parameters
    ----------
    p1, p2 : arrays
        Probability distributions.
    axis : None, int or tuple of ints
        The array dimensions along which to compute the distance.
    keepdims : bool
        The `axis` dimensions are left in the result and not squeezed out.
    **kwargs :
        Optional arguments for `np.sum`.

    Returns
    -------
    array

    Examples
    --------
    >>> bhattacharyya_distance(np.array([0.1, 0.3, 0.6]), np.array([0.2, 0.5, 0.3]))
    0.04815741684776861

    >>> bhattacharyya_distance(np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]]),
    ...                        np.array([[0.9, 0.0, 0.1], [0.2, 0.5, 0.3]]),
    ...                        axis=1)
    array([0.60706311, 0.14664722])

    """
    return -np.log(bhattacharyya_coefficient(p1, p2, **kwargs))


def hellinger_distance(p1, p2, **kwargs):
    """Compute Hellinger distance between two probability distributions.

    The Hellinger distance is computed as the square root of one minus
    the Bhattacharyya coefficient.

    See https://en.wikipedia.org/wiki/Hellinger_distance

    Parameters
    ----------
    p1, p2 : arrays
        Probability distributions.
    axis : None, int or tuple of ints
        The array dimensions along which to compute the distance.
    keepdims : bool
        The `axis` dimensions are left in the result and not squeezed out.
    **kwargs :
        Optional arguments for `np.sum`.

    Returns
    -------
    array

    Examples
    --------
    >>> hellinger_distance(np.array([0.1, 0.3, 0.6]), np.array([0.2, 0.5, 0.3]))
    0.216832286410535

    >>> hellinger_distance(np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]]),
    ...                    np.array([[0.9, 0.0, 0.1], [0.2, 0.5, 0.3]]),
    ...                    axis=1)
    array([0.6745747 , 0.36932562])

    """
    return np.sqrt(1 - bhattacharyya_coefficient(p1, p2, **kwargs))


def prob_weighted_distance(
    p1, p2, x1=None, x2=None, distances=None, exponent=2, axis=None, keepdims=False
):
    """Compute weighted distance between two probability distributions.

    Given two probability distributions that represent estimates of locations
    in a discrete space, this function will compute the mean distance between
    the two location estimates, i.e., for vectors this would amount to
    `np.sum(p1[:,None]*p2[None,:]*np.abs(x2[None,:]-x1[:,None]))`.
    In words, for every pair of locations, their distance (`np.abs(x2[j]-x1[i])`)
    is eighted by their joint probability (`p1[i]*p2[j]`).
    A `distances` matrix can be provided instead of location vectors `x1` and `x2`
    if the space is not euclidean.

    Parameters
    ----------
    p1, p2 : arrays
        Probability distributions.
    x1, x2 : None, 1d array or iterable of 1d arrays
        Location vectors associated with the probabilities in `p1` and `p2`.
        For each dimension of `p1` and `p2` (or for each dimension in `axis`
        if provided) a location vector should be given.
        `x2` can be omitted if the same as `x1`. Cannot be used together
        with `distances` argument. If `x1`, `x2` and `distances` are None,
        then location vectors are generated as np.arange(n) for each dimension.
    distances : None or array
        An array of pair-wise distances associated with the probabilities in
        `p1` and `p2`. Cannot be used together with `x1` and `x2`.
    exponent : scalar
    axis : None, int or tuple of ints
        The array dimensions along which to compute the coefficient.
    keepdims : bool
        The `axis` dimensions are left in the result and not squeezed out.

    Returns
    -------
    array

    Examples
    --------
    >>> prob_weighted_distance(np.array([0.1, 0.3, 0.6]), np.array([0.2, 0.5, 0.3]))
    1.0488088481701516

    >>> prob_weighted_distance(np.array([0.1, 0.3, 0.6]),
    ...                        np.array([0.2, 0.5, 0.3]),
    ...                        x1 = np.array([1, 3, 5]))
    2.0976176963403033

    >>> prob_weighted_distance(np.array([[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]]),
    ...                        np.array([[0.9, 0.0, 0.1], [0.2, 0.5, 0.3]]),
    ...                        axis=1)
    array([1.58113883, 1.19163753])

    """

    # p1 and p2 must have the same dimensions
    if p1.ndim != p2.ndim:
        raise ValueError("Arrays must have the same number of dimensions.")

    # axis is int, tuple of int or None
    if axis is None:
        axis = tuple(range(p1.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)

    if not all([isinstance(x, int) and x < p1.ndim and x >= -p1.ndim for x in axis]):
        raise ValueError("Invalid axis argument.")

    axis = tuple([x if x >= 0 else x + p1.ndim for x in axis])
    other_axis = [x for x in range(p1.ndim) if not x in axis]

    # p1 and p2 must have the same shape along axes not in axis
    if not all([p1.shape[x] == p2.shape[x] for x in other_axis]):
        raise ValueError(
            "Arrays must have the same shape for dimensions {}".format(other_axis)
        )

    p1_axis_shape = tuple(p1.shape[x] for x in axis)
    p2_axis_shape = tuple(p2.shape[x] for x in axis)

    nax = len(axis)

    if not distances is None and not x1 is None:
        raise ValueError("Either provide x1 and x2, or distances, but not both.")
    elif distances is None and x1 is None:
        # construct location vectors
        x1 = tuple(np.arange(n) for n in p1_axis_shape)
        x2 = (
            x1
            if p1_axis_shape == p2_axis_shape
            else tuple(np.arange(n) for n in p2_axis_shape)
        )
    elif not x1 is None:
        # check if single array => wrap in tuple
        if not isinstance(x1, (tuple, list)):
            x1 = (x1,)
        # check if len(axis) == len(p1)
        if nax != len(x1):
            raise ValueError("Expecting {} location vectors for x1.".format(nax))
        # check if all 1d arrays with len(x1[k])==p1_axis_shape[k]
        if not all([len(x) == n for x, n in zip(x1, p1_axis_shape)]):
            raise ValueError("Location vectors x1 have incorrect size.")

        if not x2 is None:
            # check if single array => wrap in tuple
            if not isinstance(x2, (tuple, list)):
                x2 = (x2,)
            # check if len(axis) == len(p2)
            if nax != len(x2):
                raise ValueError("Expecting {} location vectors for x2.".format(nax))
            # check if all 1d arrays with len(x2[k])==p2_axis_shape[k]
            if not all([len(x) == n for x, n in zip(x2, p2_axis_shape)]):
                raise ValueError("Location vectors x2 have incorrect size.")
        elif p1_axis_shape != p2_axis_shape:
            raise ValueError(
                "Please also provide x2, given that shapes for p1 and p2 are not the same"
            )

    else:  # distances is not None
        # distances must have shape p1.shape(axis) + p2.shape(axis)
        if distances.shape != p1_axis_shape + p2_axis_shape:
            raise ValueError(
                "Expecting distances array to have shape {}".format(
                    p1_axis_shape + p2_axis_shape
                )
            )

    if distances is None:
        # create distances from x1 and x2
        coords1 = np.meshgrid(*x1, indexing="ij")
        if x2 is None:
            coords2 = coords1
        else:
            coords2 = np.meshgrid(*x2, indexing="ij")

        # compute euclidean distance
        distances = 0
        for c1, c2 in zip(coords1, coords2):
            distances = distances + (
                (
                    np.expand_dims(c1, tuple(range(nax, 2 * nax)))
                    - np.expand_dims(c2, tuple(range(nax)))
                )
                ** 2
            )
        distances = np.sqrt(distances)

    p1 = np.expand_dims(
        np.moveaxis(p1, axis, tuple(range(nax))), tuple(range(nax, 2 * nax))
    )

    p2 = np.expand_dims(np.moveaxis(p2, axis, tuple(range(nax))), tuple(range(nax)))

    distances = np.expand_dims(
        distances, tuple(range(2 * nax, 2 * nax + len(other_axis)))
    )

    d = np.sum(
        p1 * p2 * (np.abs(distances) ** exponent), axis=tuple(range(2 * nax))
    ) ** (1 / exponent)

    if keepdims:
        d = np.expand_dims(d, axis=axis)

    return d
