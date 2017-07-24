from setuptools import setup
from Cython.Build import cythonize

setup(
    # ext_modules=cythonize("madness/*.pyx"),
    test_suite='nose.collector',
    tests_require=['nose'],
)
