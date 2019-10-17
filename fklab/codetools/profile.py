"""
===================================================
Profiling tools (:mod:`fklab.codetools.profile`)
===================================================

.. currentmodule:: fklab.codetools.profile

Tools for profiling code.

.. autosummary::
    :toctree: generated/

    ExecutionTimer

"""
import timeit

from fklab.version._core_version._version import __version__

__all__ = ["ExecutionTimer"]


class ExecutionTimer:
    """Context manager for timing code execution.

    Example usage:

    with ExecutionTimer() as t:
        // execute code to profile
        pass

    print("it took {} seconds".format(t.interval))

    """

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.end = timeit.default_timer()
        self.interval = self.end - self.start
