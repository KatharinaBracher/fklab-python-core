import collections

import h5py

from fklab.version._core_version._version import __version__


def map_to_hdf5(fid, data):
    """Recursively save data in a map to hdf5 file.

    Parameters
    ----------
    fid : hdf5 file node
    data :
        any data object that can be saved to hdf5 file datasets. Dictionaries
        will be traversed and keys are converted to hdf5 groups.

    """

    for k, v in data.items():
        if isinstance(v, collections.Mapping):
            group = fid.create_group(k)
            map_to_hdf5(group, v)
        else:
            try:
                fid[k] = v
            except:
                # warning or error?
                pass
