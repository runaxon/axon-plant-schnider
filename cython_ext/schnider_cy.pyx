# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
cython_ext/schnider_cy.pyx
==========================

Typed Cython implementation of the Schnider ODE hot path.

Two entry points are provided:

    derivatives_cy(state, inputs, params) -> tuple
        Drop-in replacement for physio.schnider.derivatives().
        Accepts plain Python tuples — same interface, C internals.

    derivatives_fast(double, double, double, double,
                     double, double, double, double, double, double, double)
        Pure cdef function — no Python overhead at all.
        Called directly from rk4_cy when both live in the same .pyx.

The split matters: derivatives_cy keeps the simulator's existing interface
intact so nothing else needs to change. derivatives_fast is the inner loop
that Cython can fully optimize to native arithmetic.
"""

cdef derivatives_fast(
    # state
    double x1, double x2, double x3, double xe,
    # input
    double u,
    # volume (needed for Ce ODE)
    double v1,
    # rate constants
    double k10, double k12, double k13,
    double k21, double k31,
    # effect-site
    double ke0,
):
    """
    Pure C-level Schnider ODE — no Python objects, no GIL, no boxing.
    All arguments and return values are C doubles.
    """
    cdef double dx1, dx2, dx3, dxe
    cdef double C1 = x1 / (v1 * 1000.0)   # µg/mL

    dx1 = u - (k10 + k12 + k13) * x1 + k21 * x2 + k31 * x3
    dx2 = k12 * x1 - k21 * x2
    dx3 = k13 * x1 - k31 * x3
    dxe = ke0 * (C1 - xe)                  # xe is Ce (µg/mL), correct ODE

    return dx1, dx2, dx3, dxe


def derivatives_cy(tuple state, tuple inputs, tuple params):
    """
    Drop-in replacement for physio.schnider.derivatives().

    Accepts the same plain Python tuples as the original — the interface
    is unchanged.  Unpacking happens once at the Python/C boundary, then
    all arithmetic runs as typed C doubles via derivatives_fast().
    """
    cdef double x1, x2, x3, xe
    cdef double u
    cdef double v1, k10, k12, k13, k21, k31, ke0

    x1, x2, x3, xe = state
    (u,)            = inputs
    v1  = params[0]
    k10 = params[3]
    k12 = params[4]
    k13 = params[5]
    k21 = params[6]
    k31 = params[7]
    ke0 = params[8]

    return derivatives_fast(x1, x2, x3, xe, u, v1, k10, k12, k13, k21, k31, ke0)
