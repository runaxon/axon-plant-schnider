"""
Act 2: Typed Cython build for the derivatives() hot path.

Compiles only schnider_cy.pyx — the single function that accounts for
the vast majority of simulation runtime. Everything else stays pure Python.

Build:
    python cython_ext/setup_typed.py build_ext --inplace

Produces: cython_ext/schnider_cy.<platform>.so
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        'cython_ext/schnider_cy.pyx',
        compiler_directives={
            'language_level': '3',
            'boundscheck':    False,
            'wraparound':     False,
            'cdivision':      True,
        },
        annotate=True,
    )
)
