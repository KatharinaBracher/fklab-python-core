import warnings

from ..styles import *
from fklab.codetools import FKLabDeprecationWarning

warnings.warn(
    "The fklab.plot.core.styles module is deprecated. "
    "Please use fklab.plot.styles instead.",
    FKLabDeprecationWarning,
    stacklevel=2,
)
