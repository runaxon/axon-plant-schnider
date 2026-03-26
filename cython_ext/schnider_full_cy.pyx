# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
cython_ext/schnider_full_cy.pyx
================================

Full C hot path: ODE derivatives, RK4 integrator, Hill equation (BIS),
and PID controller are all cdef functions — no Python boundary inside the
simulation loop.

Entry points
------------
    simulate_closed_loop_cy(patient, kp, ki, kd, duration, dt)
        Grid-search entry point. Takes PID gains directly, returns only the
        scalar ISE loss. No time-series allocation — maximum throughput.

    simulate_closed_loop_cy_full(patient, kp, ki, kd, duration, dt)
        Same physics, but collects the full time-series into a SimulationResult
        for plotting and diagnostics.
"""

from libc.math cimport pow as cpow

from physio.core import SimulationResult


# ---------------------------------------------------------------------------
# PID controller state — plain C struct, no Python objects
# ---------------------------------------------------------------------------

cdef struct PIDState:
    double kp
    double ki
    double kd
    double setpoint
    double dt
    double max_rate
    double min_rate
    double integral
    double prev_measurement
    int    first_step          # 1 on the first call (no derivative yet)


cdef inline double pid_step(PIDState *pid, double measurement) noexcept nogil:
    cdef double error, p, d, i_term, output
    cdef int saturated

    error = measurement - pid.setpoint

    p = pid.kp * error

    if pid.first_step:
        d = 0.0
        pid.first_step = 0
    else:
        d = pid.kd * (measurement - pid.prev_measurement) / pid.dt

    i_term = pid.ki * pid.integral

    output = p + i_term + d

    saturated = (output >= pid.max_rate) or (output <= pid.min_rate)
    if not saturated:
        pid.integral += error * pid.dt

    pid.prev_measurement = measurement

    if output > pid.max_rate:
        return pid.max_rate
    if output < pid.min_rate:
        return pid.min_rate
    return output


# ---------------------------------------------------------------------------
# Schnider ODE derivatives — pure C, pointer outputs
# ---------------------------------------------------------------------------

cdef inline void derivatives_fast(
    double x1, double x2, double x3, double xe,
    double u,
    double k10, double k12, double k13, double k21, double k31, double ke0,
    double *dx1, double *dx2, double *dx3, double *dxe,
) noexcept nogil:
    dx1[0] = u - (k10 + k12 + k13) * x1 + k21 * x2 + k31 * x3
    dx2[0] = k12 * x1 - k21 * x2
    dx3[0] = k13 * x1 - k31 * x3
    dxe[0] = ke0 * (x1 - xe)


# ---------------------------------------------------------------------------
# RK4 integrator — calls derivatives_fast directly, four times, zero Python
# ---------------------------------------------------------------------------

cdef inline void rk4_step_cy(
    double x1, double x2, double x3, double xe,
    double u,
    double k10, double k12, double k13, double k21, double k31, double ke0,
    double dt,
    double *ox1, double *ox2, double *ox3, double *oxe,
) noexcept nogil:
    cdef double k1x1, k1x2, k1x3, k1xe
    cdef double k2x1, k2x2, k2x3, k2xe
    cdef double k3x1, k3x2, k3x3, k3xe
    cdef double k4x1, k4x2, k4x3, k4xe
    cdef double s2x1, s2x2, s2x3, s2xe
    cdef double s3x1, s3x2, s3x3, s3xe
    cdef double s4x1, s4x2, s4x3, s4xe
    cdef double h2  = 0.5 * dt
    cdef double dt6 = dt / 6.0

    derivatives_fast(x1,   x2,   x3,   xe,   u, k10,k12,k13,k21,k31,ke0,
                     &k1x1,&k1x2,&k1x3,&k1xe)

    s2x1 = x1 + h2*k1x1;  s2x2 = x2 + h2*k1x2
    s2x3 = x3 + h2*k1x3;  s2xe = xe + h2*k1xe

    derivatives_fast(s2x1,s2x2,s2x3,s2xe, u, k10,k12,k13,k21,k31,ke0,
                     &k2x1,&k2x2,&k2x3,&k2xe)

    s3x1 = x1 + h2*k2x1;  s3x2 = x2 + h2*k2x2
    s3x3 = x3 + h2*k2x3;  s3xe = xe + h2*k2xe

    derivatives_fast(s3x1,s3x2,s3x3,s3xe, u, k10,k12,k13,k21,k31,ke0,
                     &k3x1,&k3x2,&k3x3,&k3xe)

    s4x1 = x1 + dt*k3x1;  s4x2 = x2 + dt*k3x2
    s4x3 = x3 + dt*k3x3;  s4xe = xe + dt*k3xe

    derivatives_fast(s4x1,s4x2,s4x3,s4xe, u, k10,k12,k13,k21,k31,ke0,
                     &k4x1,&k4x2,&k4x3,&k4xe)

    ox1[0] = x1 + dt6*(k1x1 + 2.0*k2x1 + 2.0*k3x1 + k4x1)
    ox2[0] = x2 + dt6*(k1x2 + 2.0*k2x2 + 2.0*k3x2 + k4x2)
    ox3[0] = x3 + dt6*(k1x3 + 2.0*k2x3 + 2.0*k3x3 + k4x3)
    oxe[0] = xe + dt6*(k1xe + 2.0*k2xe + 2.0*k3xe + k4xe)


# ---------------------------------------------------------------------------
# Hill equation (BIS) — pure C
# ---------------------------------------------------------------------------

cdef inline double bis_fast(
    double xe,
    double v1,
    double e0, double emax, double ec50, double gamma,
) noexcept nogil:
    cdef double ce
    ce = xe / (v1 * 1000.0)
    if ce <= 0.0:
        return e0
    cdef double ceg = cpow(ce, gamma)
    cdef double val = e0 - emax * ceg / (cpow(ec50, gamma) + ceg)
    if val < 0.0:
        return 0.0
    return val


# ---------------------------------------------------------------------------
# Grid-search entry point — returns scalar ISE loss, no time-series
# ---------------------------------------------------------------------------

def simulate_closed_loop_cy(
    patient,
    double kp, double ki, double kd,
    double duration=60.0,
    double dt=0.1,
    double bis_target=50.0,
    double t_induction=2.0,
    double t_maintenance=30.0,
):
    """
    Run one closed-loop simulation and return the ISE loss scalar.

    Designed for grid search — skips time-series allocation entirely.
    Everything inside the loop runs in C: ODE, RK4, Hill equation, PID.
    """
    from physio.schnider import params_from_patient, STATE0

    cdef double x1, x2, x3, xe
    cdef double ox1, ox2, ox3, oxe
    cdef double k10, k12, k13, k21, k31, ke0
    cdef double v1, e0, emax, ec50, gamma
    cdef double u, bis
    cdef double t, ise
    cdef int    n_steps, i
    cdef PIDState pid

    params = params_from_patient(patient)

    v1    = params[0]
    k10   = params[3];  k12 = params[4];  k13 = params[5]
    k21   = params[6];  k31 = params[7];  ke0 = params[8]
    e0    = params[9];  emax = params[10]; ec50 = params[11]; gamma = params[12]

    x1, x2, x3, xe = STATE0

    pid.kp             = kp
    pid.ki             = ki
    pid.kd             = kd
    pid.setpoint       = bis_target
    pid.dt             = dt
    pid.max_rate       = 300_000.0
    pid.min_rate       = 0.0
    pid.integral       = 0.0
    pid.prev_measurement = 0.0
    pid.first_step     = 1

    t       = 0.0
    ise     = 0.0
    n_steps = int(round(duration / dt))

    for i in range(n_steps):
        bis = bis_fast(xe, v1, e0, emax, ec50, gamma)
        u   = pid_step(&pid, bis)

        # Accumulate ISE only during the maintenance window
        if t >= t_induction and t <= t_maintenance:
            ise += (bis - bis_target) * (bis - bis_target) * dt

        rk4_step_cy(x1, x2, x3, xe, u, k10, k12, k13, k21, k31, ke0, dt,
                    &ox1, &ox2, &ox3, &oxe)
        x1 = ox1;  x2 = ox2;  x3 = ox3;  xe = oxe
        t += dt

    return ise


# ---------------------------------------------------------------------------
# Full time-series version — for plotting and diagnostics
# ---------------------------------------------------------------------------

def simulate_closed_loop_cy_full(
    patient,
    double kp, double ki, double kd,
    double duration=60.0,
    double dt=0.1,
    double bis_target=50.0,
):
    """
    Same physics as simulate_closed_loop_cy but collects the full time-series.
    Drop-in replacement for grid_search.simulate_closed_loop() for plotting.
    """
    from physio.schnider import params_from_patient, STATE0

    cdef double x1, x2, x3, xe
    cdef double ox1, ox2, ox3, oxe
    cdef double k10, k12, k13, k21, k31, ke0
    cdef double v1, e0, emax, ec50, gamma
    cdef double u, bis, cp, ce
    cdef double t
    cdef int    n_steps, i
    cdef PIDState pid

    params = params_from_patient(patient)

    v1    = params[0]
    k10   = params[3];  k12 = params[4];  k13 = params[5]
    k21   = params[6];  k31 = params[7];  ke0 = params[8]
    e0    = params[9];  emax = params[10]; ec50 = params[11]; gamma = params[12]

    x1, x2, x3, xe = STATE0

    pid.kp             = kp
    pid.ki             = ki
    pid.kd             = kd
    pid.setpoint       = bis_target
    pid.dt             = dt
    pid.max_rate       = 300_000.0
    pid.min_rate       = 0.0
    pid.integral       = 0.0
    pid.prev_measurement = 0.0
    pid.first_step     = 1

    t       = 0.0
    n_steps = int(round(duration / dt))

    result = SimulationResult()

    for i in range(n_steps):
        bis = bis_fast(xe, v1, e0, emax, ec50, gamma)
        cp  = x1 / (v1 * 1000.0)
        ce  = xe / (v1 * 1000.0)
        u   = pid_step(&pid, bis)

        result.time.append(t)
        result.outputs.setdefault('bis',  []).append(bis)
        result.outputs.setdefault('cp',   []).append(cp)
        result.outputs.setdefault('ce',   []).append(ce)
        result.outputs.setdefault('rate', []).append(u)

        rk4_step_cy(x1, x2, x3, xe, u, k10, k12, k13, k21, k31, ke0, dt,
                    &ox1, &ox2, &ox3, &oxe)
        x1 = ox1;  x2 = ox2;  x3 = ox3;  xe = oxe
        t += dt

    # Final point
    bis = bis_fast(xe, v1, e0, emax, ec50, gamma)
    cp  = x1 / (v1 * 1000.0)
    ce  = xe / (v1 * 1000.0)
    u   = pid_step(&pid, bis)
    result.time.append(t)
    result.outputs.setdefault('bis',  []).append(bis)
    result.outputs.setdefault('cp',   []).append(cp)
    result.outputs.setdefault('ce',   []).append(ce)
    result.outputs.setdefault('rate', []).append(u)

    return result
