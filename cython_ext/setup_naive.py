"""
Act 1: Naive Cython compilation of schnider.py with zero changes.

This compiles physio/schnider.py as-is using Cython — no type annotations,
no cdef, no typed memoryviews. The result is essentially Python bytecode
compiled to C, which gives a modest speedup from removing interpreter
overhead but does not eliminate the dynamic typing cost.

Build:
    python cython_ext/setup_naive.py build_ext --inplace

This produces a .so next to schnider.py that Python will import in place
of the pure Python version.
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        'physio/schnider.py',
        compiler_directives={'language_level': '3'},
        annotate=True,    # generates schnider.html showing Python vs C lines
    )
)
