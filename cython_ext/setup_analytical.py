from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        'cython_ext/schnider_analytical_cy.pyx',
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        },
    )
)
