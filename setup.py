from setuptools import setup
from distutils.core import Extension
import sys

# because we have namespace packages without __init__.py
# which are not detected automatically by find_packages()
# we need to explicitly specify the packages
packages = [
    "fklab.version._core_version",
    "fklab.codetools",
    "fklab.events",
    "fklab.segments",
    "fklab.geometry",
    "fklab.io",
    "fklab.io.common",
    "fklab.io.mwl",
    "fklab.io.neuralynx",
    "fklab.io.openephys",
    "fklab.plot",
    "fklab.plot.plots",
    "fklab.plot.core",
    "fklab.plot.neuralynx",
    "fklab.plot.openephys",
    "fklab.radon",
    "fklab.signals.core",
    "fklab.signals.filter",
    "fklab.signals.smooth",
    "fklab.signals.multirate",
    "fklab.signals.multitaper",
    "fklab.statistics.core",
    "fklab.statistics.circular",
    "fklab.statistics.correlation",
    "fklab.statistics.distances",
    "fklab.statistics.information",
    "fklab.utilities",
]


class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __iter__(self):
        for k in str(self):
            yield k

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)

if sys.platform.startswith('win'):
    compile_args = ['/std:c++17', '/O2']
elif sys.platform.startswith('linux'):
    compile_args = ['-std=c++17', '-O3']
else:
    compile_args = []

radon_ext = Extension(
    "fklab.radon.radonc",
    sources=["fklab/radon/src/pybind_radon.cpp", "fklab/radon/src/radon.cpp"],
    libraries=[],
    include_dirs=[get_pybind_include(), get_pybind_include(user=True)],
    language="c++",
    extra_compile_args=compile_args,
)


import re

VERSIONFILE = "fklab/version/_core_version/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name="fklab-python-core",
    version=verstr,
    packages=packages,
    ext_modules=[radon_ext],
    install_requires=[
        "numpy>=1.17,<=1.25",
        "scipy>=1.2",
        "numba",
        "h5py",
        "pybind11>=2.2",
        "matplotlib",
        "pyyaml",
        "scikit-learn",
        "typing_extensions",
        "pandas"
    ],
    author="Fabian Kloosterman",
    author_email="fabian.kloosterman@nerf.be",
    description="Kloosterman Lab Data Analysis Tools",
    license="GPL3",
    zip_safe=False,
    include_package_data=True,
)
