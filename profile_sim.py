"""
Profiling script for the Schnider simulation.

Two levels of profiling:

    1. cProfile  — broad picture: which functions consume the most time
    2. line_profiler — surgical view: which lines inside rk4_step are slow

Usage
-----
    # Level 1 — cProfile
    python profile_sim.py

    # Level 2 — line_profiler (requires: pip install line_profiler)
    kernprof -l -v profile_sim.py
"""

import cProfile
import pstats
import io

from physio.cohort import cohort_from_json
from physio.schnider import simulate

# Small cohort — enough to get stable profile numbers, fast enough to be interactive
COHORT   = cohort_from_json('cohorts/n200_seed99.json')[:20]
SCHEDULE = [
    (0.0,  140_000.0),
    (1.0,    7_000.0),
    (30.0,       0.0),
]
DURATION = 60.0
DT       = 0.1


def run(cohort=COHORT):
    """The workload being profiled — simulate a cohort."""
    return [simulate(p, SCHEDULE, duration=DURATION, dt=DT) for p in cohort]


# ---------------------------------------------------------------------------
# Level 2: line_profiler — decorate the hot functions
# Kernprof will instrument these when run with `kernprof -l -v profile_sim.py`
# When run normally, @profile is a no-op passthrough.
# ---------------------------------------------------------------------------
try:
    profile  # injected by kernprof
except NameError:
    def profile(fn):
        return fn


@profile
def rk4_step_profiled(deriv, state, inputs, params, dt):
    """
    Inline copy of physio.core.rk4_step — decorated so kernprof can show
    which lines are slow at a granular level.
    """
    k1 = deriv(state, inputs, params)

    s2 = tuple(s + 0.5 * dt * d for s, d in zip(state, k1))
    k2 = deriv(s2, inputs, params)

    s3 = tuple(s + 0.5 * dt * d for s, d in zip(state, k2))
    k3 = deriv(s3, inputs, params)

    s4 = tuple(s + dt * d for s, d in zip(state, k3))
    k4 = deriv(s4, inputs, params)

    return tuple(
        s + (dt / 6.0) * (d1 + 2*d2 + 2*d3 + d4)
        for s, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4)
    )


@profile
def derivatives_profiled(state, inputs, params):
    """
    Inline copy of physio.schnider.derivatives — decorated so kernprof can
    show which lines are slow at a granular level.
    """
    x1, x2, x3, xe = state
    (u,) = inputs
    v1, v2, v3, k10, k12, k13, k21, k31, ke0 = params[:9]

    dx1 = u - (k10 + k12 + k13) * x1 + k21 * x2 + k31 * x3
    dx2 = k12 * x1 - k21 * x2
    dx3 = k13 * x1 - k31 * x3
    dxe = ke0 * (x1 - xe)

    return (dx1, dx2, dx3, dxe)


def run_profiled(cohort=COHORT):
    """Same workload but using the @profile-decorated hot functions."""
    import physio.core as _core_mod
    from physio.schnider import params_from_patient, STATE0, outputs
    from physio.core import simulate as core_simulate
    # Temporarily swap in the profiled rk4_step so the simulate loop calls it
    _orig = _core_mod.rk4_step
    _core_mod.rk4_step = rk4_step_profiled
    try:
        result = [
            core_simulate(
                derivatives_profiled,
                outputs,
                STATE0,
                params_from_patient(p),
                SCHEDULE,
                duration=DURATION,
                dt=DT,
            )
            for p in cohort
        ]
    finally:
        _core_mod.rk4_step = _orig
    return result


# ---------------------------------------------------------------------------
# Level 1: cProfile
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 60)
    print('Level 1: cProfile — cumulative time per function')
    print('=' * 60)

    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(20)
    print(stream.getvalue())

    print()
    print('=' * 60)
    print('Level 2: line_profiler — run with:')
    print('  pip install line_profiler')
    print('  kernprof -l -v profile_sim.py')
    print('=' * 60)

# Always call run_profiled so kernprof can instrument the @profile functions
# (kernprof only collects data for functions that are actually called)
run_profiled()
