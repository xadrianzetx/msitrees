import os
from sysconfig import get_paths
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


__version__ = '0.1.dev0'


def get_long_desc():
    # TODO
    pass


def include_numpy():
    """Get path to numpy headers after setup ensured install"""
    import numpy as np
    return np.get_include()


def include_pybind():
    """Get path to pybind headers after setup ensured install"""
    import pybind11
    return pybind11.get_include()


def get_ext_modules():
    """Get external modules to compile"""
    ext = Extension(
        name='msitrees._core',
        sources=[os.path.join('msitrees', 'core', 'core.cpp')],
        include_dirs=[
            get_paths()['include'],
            include_numpy(),
            include_pybind(),
            os.path.join(os.getcwd(), 'msitrees', 'core', 'include')
        ],
        language='c++'
    )
    return [ext]


setup(
    name='msitrees',
    version=__version__,
    author='xadrianzetx',
    packages=find_packages(),
    ext_modules=get_ext_modules(),
    python_requires='>=3.5',
    setup_requires=[
        'pybind11>=2.5.0',
        'numpy',
        'wheel'
    ],
    cmdclass={'build_ext': build_ext},
    zip_safe=False
)
