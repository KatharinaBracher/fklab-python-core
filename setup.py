
from setuptools import setup
from distutils.core import Extension

# because we have namespace packages without __init__.py
# which are not detected automatically by find_packages()
# we need to explicitly specify the packages
packages = [ 'fklab.codetools',
             'fklab.events',
             'fklab.segments',
             'fklab.io',
             'fklab.plot',
             'fklab.radon',
             'fklab.signals',
             'fklab.statistics',
             'fklab.utilities',
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

radon_ext = Extension('fklab.radon.radonc',
                sources = ['fklab/radon/src/pybind_radon.cpp', 'fklab/radon/src/radon.cpp'],
                libraries = [],
                include_dirs = [get_pybind_include(), get_pybind_include(user=True)],
                language = "c++",
                extra_compile_args = ['-std=c++11', '-O3']
               )


setup(
    name="fklab",
    version="1.0",
    packages=packages,
    
    ext_modules = [radon_ext,],
    
    install_requires=['python>=3.6', 'numpy', 'scipy', 'numba', 'h5py', 'pybind11', 'matplotlib', 'spectrum', 'enum'],

    author="Fabian Kloosterman",
    author_email="fabian.kloosterman@nerf.be",
    description="Kloosterman Lab Data Analysis Tools",
    license="GPL3",
    
    zip_safe=False,
    include_package_data=True,
    
)

