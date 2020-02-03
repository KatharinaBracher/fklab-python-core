"""
======================================================
Neuralynx (:mod:`fklab.io.neuralynx.Nlx2Kilosort`)
======================================================

.. currentmodule:: fklab.io.neuralynx

Utilities collection for Neuralynx acquisition system

.. autosummary::
    :toctree: generated/

        data_generation
        tetrode

"""

__all__ = ["data_generation", "tetrode"]


import os

import numpy as np
from fklab.io.template_probe import template_probe
from fklab.io.neuralynx import NlxOpen
from fklab.utilities.general import natural_sort
from fklab.utilities.general import slices

from fklab.version._internal_version._version import __version__


def data_generation(
    name, destination, file_names=None, sort="natural", time_window=None
):
    """ generate .dat file for kilosort from CSC files

    Parameters
    ----------

    name: str
          name of the output file

    destination: str
          path to the folder where the output file will be saved

    file_names: str, opt
          path to the CSC files of interest

    sort: str, opt
          {'natural', 'alphabet', 'arbitrary'} Define in which order the files are sorted, natural or alphabetical. Default is natural.

    time_window: list, opt
          [start, end] time

    """
    if sort == "natural":
        filenames = natural_sort(file_names)
    elif sort == "alphabet":
        filenames = file_names.sort()
    elif sort == "arbitrary":
        filenames = file_names
    else:
        raise (
            ValueError(
                "Wrong sort option: there are only 3 choices : natural, alphabet, arbitrary"
            )
        )

    # open all files
    files = [NlxOpen(f, scale_data=False) for f in filenames]

    if time_window is None:
        tim_window = [files[0].starttime, files[0].endtime]

    s = files[0].timeslice(start=time_window[0], stop=time_window[1])
    start_index, end_index = s.start, s.stop
    nrecords = end_index - start_index + 1
    print(nrecords)

    # create output dat file
    with open(os.path.join(destination, name + ".dat"), "wb") as destination_file:

        for selection in slices(nrecords, size=10000, start=start_index):
            signals = [
                np.reshape(f.data.signal_by_sample[selection], (-1, 1)) for f in files
            ]
            signals = np.hstack(signals).astype("int16")
            signals.tofile(destination_file, sep="")


class tetrode(template_probe):
    def __init__(self, n_tetrodes, spacing=200):
        super().__init__()
        self.n_tetrodes = n_tetrodes
        self.spacing = spacing

    def create_channel_map(self):

        """ generate channelmap files for Kilosort
            for tetrodes recordings,
            each tetrode channels is positionned on a single row

        Parameters
        ----------

        name: str
            name of the output file

        destination: str
            path to the folder where the output file will be saved

        n_tetrodes: scalar
            number of tetrodes to generate the map for

        spacing: int, opt
            space between each row (ie tetrode) on the channel map

        """

        self._chanMap = np.arange(self.n_tetrodes * 4) + 1
        self._chanMap0ind = np.arange(self.n_tetrodes * 4)
        self._connected = np.zeros(len(self._chanMap))
        self._xCoord = np.array([0, 5, 10, 15] * self.n_tetrodes)
        self._yCoord = np.repeat(np.arange(self.n_tetrodes) * self.spacing, 4)
        self._kCoord = np.zeros(len(self._chanMap))
        self._elecInd = np.arange(1, len(self._chanMap))
