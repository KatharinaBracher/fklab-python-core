"""
=============================================
OpenEphys File IO (:mod:`fklab.io.openephys`)
=============================================

.. currentmodule:: fklab.io.openephys

OpenEphys file import functions.


"""
from .openephys import *

__all__ = [s for s in dir() if not s.startswith("_")]
