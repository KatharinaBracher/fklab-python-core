"""


.. currentmodule:: fklab.utilities.yaml

This module will import the complete namespace of the PyYaml package and
add support for OrderedDicts.

"""
from collections import OrderedDict as _OrderedDict

from yaml import *

from fklab.version._core_version._version import __version__

# note that we will use the default mapping tag,
# which means that all maps are loaded as OrderedDicts
_mapping_tag = resolver.BaseResolver.DEFAULT_MAPPING_TAG


def dict_representer(dumper, data):
    return dumper.represent_mapping(_mapping_tag, iter(data.items()))


def dict_constructor(loader, node):
    return _OrderedDict(loader.construct_pairs(node))


add_representer(_OrderedDict, dict_representer)
add_constructor(_mapping_tag, dict_constructor)
