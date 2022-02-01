import warnings

from ..colors import *
from fklab.codetools import FKLabDeprecationWarning

warnings.warn(
    "The fklab.plot.core.colors module is deprecated. "
    "Please use fklab.plot.colors instead.",
    FKLabDeprecationWarning,
    stacklevel=2,
)
