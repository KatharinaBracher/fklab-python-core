"""
=============================================================
Information (:mod:`fklab.statistics.information.information`)
=============================================================

.. currentmodule:: fklab.statistics.information.information

Information theoretic functions.
"""
import numpy as np

from fklab.version._core_version._version import __version__

__all__ = ["spatial_information"]


def spatial_information(spike_rate, occupancy, units="bits/spike"):
    """Compute spatial information measure.

    Parameters
    ----------
    spike_rate : array or sequence of arrays
    occupancy : array
    units : {'bits/spike','bits/second'}

    Returns
    -------
    scalar or list of scalars

    """
    if not isinstance(spike_rate, (list, tuple)):
        spike_rate = [spike_rate]

    spike_rate = [np.array(x, copy=False) for x in spike_rate]

    occupancy = np.array(occupancy, copy=False)

    if not all([x.shape == occupancy.shape for x in spike_rate]):
        raise ValueError("Arrays do not have same shape")

    if units not in ("bits/spike", "bits/second"):
        raise ValueError("Invalid units. Should be bits/spike or bits/second.")

    nspikes = [x * occupancy for x in spike_rate]
    totalspikes = [np.nansum(x) for x in nspikes]
    totaltime = np.nansum(occupancy)

    si = [
        np.nansum((y / z) * np.log2(x / (z / totaltime)))
        for x, y, z in zip(spike_rate, nspikes, totalspikes)
    ]

    if units == "bits/second":
        si = [x * y / totaltime for x, y in zip(si, totalspikes)]

    return si
