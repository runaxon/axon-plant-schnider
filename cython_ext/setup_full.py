"""
Act 3: Full C hot path build.

Compiles schnider_full_cy.pyx — both derivatives() and rk4_step() as cdef
functions in the same translation unit, so the RK4 loop calls derivatives
with zero Python overhead.

Build:
    python cython_ext/setup_full.py build_ext --inplace

Produces: cython_ext/schnider_full_cy.<platform>.so
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        'cython_ext/schnider_full_cy.pyx',
        compiler_directives={
            'language_level': '3',
            'boundscheck':    False,
            'wraparound':     False,
            'cdivision':      True,
        },
        annotate=True,
    )
)
