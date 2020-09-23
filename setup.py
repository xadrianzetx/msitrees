import os
from sysconfig import get_paths
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext


__version__ = '0.2.dev1'


def get_long_desc():
    with open('README.md', 'r') as file:
        description = file.read()
    return description


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
        language='c++',
        # extra flags for linux dist
        # https://github.com/xtensor-stack/xtensor/issues/1704
        # on windows -ffast-math is enabled by default with
        # equivalent /fp:precise flag
        extra_link_args=['-lstdc++'],
        extra_compile_args=['-std=c++14', '-ffast-math', '-mavx2']
    )
    return [ext]


setup(
    name='msitrees',
    version=__version__,
    author='xadrianzetx',
    url='https://github.com/xadrianzetx/msitrees',
    description='MSI based machine learning algorithms',
    long_description=get_long_desc(),
    long_description_content_type='text/markdown',
    project_urls={
        'Documentation': 'https://msitrees.readthedocs.io/en/latest/index.html',
        'Source Code': 'https://github.com/xadrianzetx/msitrees/tree/master'
    },
    packages=find_packages(include=['msitrees']),
    ext_modules=get_ext_modules(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries'
    ],
    python_requires='>=3.5',
    install_requires=[
        'numpy>=1.18',
        'pandas>=1.0.0',
        'joblib>=0.16.0'
    ],
    cmdclass={'build_ext': build_ext},
    zip_safe=False
)
