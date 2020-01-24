"""
======================================================
Neuralynx (:mod:`fklab.io.neuralynx.Nlx2Kilosort`)
======================================================

.. currentmodule:: fklab.io.neuralynx

Utilities collection for Neuralynx acquisition system

.. autosummary::
    :toctree: generated/

        data_generation
        channel_map_generation

"""

__all__ = ["data_generation", "channel_map_generation"]


import os

import numpy as np

from fklab.io.neuralynx import NlxOpen
from fklab.utilities.general import natural_sort
from fklab.utilities.general import slices
from fklab.version._internal_version._version import __version__

##TO DO: Update generation to be in numpy format


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


def channel_map_generation(name, destination, n_tetrodes, spacing=200):

    """ generate channelmap files for Kilosort
        for tetrodes recordings,
        each tetrode channels is positionned on a single row
        #TO DO: remove the '#' in the file's header
                find a solution for tetrodes that have less than 4 channels

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

    probe = np.arange(n_tetrodes * 4) + 1
    data = np.arange(n_tetrodes * 4)
    disconnect = np.zeros(len(probe))
    x = np.array([0, 5, 10, 15] * n_tetrodes)
    y = np.repeat(np.arange(n_tetrodes) * spacing, 4)
    shank = np.zeros(len(probe))

    mapping = (
        np.vstack((probe, data, disconnect, x, y, shank)).swapaxes(0, 1).astype(int)
    )

    np.savetxt(
        os.path.join(destination, name + ".channelmap"),
        mapping,
        fmt="%1i",
        delimiter=",",
        header="Probe,Data,Disconnect,X,Y,Shank",
    )
