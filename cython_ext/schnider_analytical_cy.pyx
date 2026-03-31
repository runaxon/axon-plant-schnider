# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
cython_ext/schnider_analytical_cy.pyx
======================================

C extension for the closed-form analytical solution of the Schnider model.

Each simulation step is:
    x[n+1] = Ad * x[n] + Bd * u[n]         (9 multiplies, 6 adds)
    Ce[n+1] = alpha*Ce[n] + sum_i h_i*c_i + sum_i p_i*u   (precomputed)

Ad (3x3), Bd (3,), and all Ce coefficients are precomputed once per
(patient, dt) pair at Python boundary and passed as flat C arrays.

Entry points
------------
    simulate_closed_loop_analytical(patient, kp, ki, kd, Ad, Bd,
                                    alpha, h_coeffs, p_coeffs,
                                    Vinv, lam, duration, dt)
        Grid-search entry point — returns scalar ISE.

    precompute_matrices(params, dt)
        Computes Ad, Bd, alpha, h_coeffs, p_coeffs, Vinv, lam from Python.
        Call once per patient, pass results to simulate_closed_loop_analytical.
"""

from libc.math cimport pow as cpow, exp as cexp
import numpy as np
from scipy.linalg import expm

from physio.core import SimulationResult


# ---------------------------------------------------------------------------
# PID controller — same as schnider_full_cy.pyx
# ---------------------------------------------------------------------------

cdef struct PIDState:
    double kp, ki, kd
    double setpoint, dt
    double max_rate, min_rate
    double integral, prev_measurement
    int    first_step

cdef inline double pid_step(PIDState *pid, double measurement) noexcept nogil:
    cdef double error, p, d, i_term, output
    error  = measurement - pid.setpoint
    p      = pid.kp * error
    d      = 0.0 if pid.first_step else pid.kd * (measurement - pid.prev_measurement) / pid.dt
    i_term = pid.ki * pid.integral
    output = p + i_term + d
    if not (output >= pid.max_rate or output <= pid.min_rate):
        pid.integral += error * pid.dt
    pid.prev_measurement = measurement
    pid.first_step = 0
    if output > pid.max_rate: return pid.max_rate
    if output < pid.min_rate: return pid.min_rate
    return output


# ---------------------------------------------------------------------------
# Hill equation (BIS)
# ---------------------------------------------------------------------------

cdef inline double bis_fast(
    double ce,
    double e0, double emax, double ec50, double gamma,
) noexcept nogil:
    if ce <= 0.0:
        return e0
    cdef double ceg = cpow(ce, gamma)
    cdef double val = e0 - emax * ceg / (cpow(ec50, gamma) + ceg)
    return val if val > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Analytical state update — x[n+1] = Ad*x[n] + Bd*u
# Ad is stored row-major as a flat 9-element array
# ---------------------------------------------------------------------------

cdef inline void analytical_step(
    double x1, double x2, double x3,
    double u,
    double *Ad,   # 3x3 row-major
    double *Bd,   # 3-vector
    double *ox1, double *ox2, double *ox3,
) noexcept nogil:
    ox1[0] = Ad[0]*x1 + Ad[1]*x2 + Ad[2]*x3 + Bd[0]*u
    ox2[0] = Ad[3]*x1 + Ad[4]*x2 + Ad[5]*x3 + Bd[1]*u
    ox3[0] = Ad[6]*x1 + Ad[7]*x2 + Ad[8]*x3 + Bd[2]*u


# ---------------------------------------------------------------------------
# Ce update — exact closed form
# Ce[n+1] = alpha*Ce[n] + sum_i (h_i*c_i + p_i*u)
# where c_i = Vinv[i,0]*x1 + Vinv[i,1]*x2 + Vinv[i,2]*x3  (modal coords)
# ---------------------------------------------------------------------------

cdef inline double ce_step(
    double x1, double x2, double x3,
    double Ce,
    double u,
    double alpha,
    double *h_coeffs,   # 3-vector: ke0*alpha*scale*V0[i]*I1[i]
    double *p_coeffs,   # 3-vector: ke0*alpha*scale*V0[i]*d[i]/lam[i]*(I2[i]-I3)
    double *Vinv,       # 3x3 row-major
) noexcept nogil:
    cdef double Ce_new, c0, c1, c2

    # Modal coordinates: c = Vinv @ x
    c0 = Vinv[0]*x1 + Vinv[1]*x2 + Vinv[2]*x3
    c1 = Vinv[3]*x1 + Vinv[4]*x2 + Vinv[5]*x3
    c2 = Vinv[6]*x1 + Vinv[7]*x2 + Vinv[8]*x3

    Ce_new = (alpha * Ce
              + h_coeffs[0] * c0 + h_coeffs[1] * c1 + h_coeffs[2] * c2
              + (p_coeffs[0] + p_coeffs[1] + p_coeffs[2]) * u)
    return Ce_new


# ---------------------------------------------------------------------------
# Python-level precomputation — call once per (patient, dt)
# ---------------------------------------------------------------------------

def precompute_matrices(params, double dt):
    """
    Precompute Ad, Bd, and Ce update coefficients from params and dt.

    Returns a dict with all precomputed values as numpy arrays.
    Call once per patient; pass results to simulate_closed_loop_analytical.
    """
    v1, v2, v3, k10, k12, k13, k21, k31, ke0 = params[:9]

    A = np.array([
        [-(k10 + k12 + k13),  k21,  k31],
        [ k12,               -k21,    0],
        [ k13,                  0, -k31],
    ], dtype=np.float64)
    B = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    Ad = expm(A * dt)
    Bd = np.linalg.solve(A, (Ad - np.eye(3))) @ B

    eigenvalues, V = np.linalg.eig(A)
    Vinv = np.linalg.inv(V)
    lam  = eigenvalues.real
    V0   = V[0, :].real
    d    = (Vinv @ B).real
    scale = 1.0 / (v1 * 1000.0)

    alpha = np.exp(-ke0 * dt)
    I3    = (np.exp(ke0 * dt) - 1.0) / ke0

    h_coeffs = np.zeros(3)
    p_coeffs = np.zeros(3)
    for i in range(3):
        li = lam[i]
        I1 = dt if abs(li + ke0) < 1e-12 else (np.exp((li + ke0) * dt) - 1.0) / (li + ke0)
        h_coeffs[i] = ke0 * alpha * scale * V0[i] * I1
        p_coeffs[i] = ke0 * alpha * scale * V0[i] * d[i] / li * (I1 - I3)

    return {
        'Ad':       np.ascontiguousarray(Ad.flatten(), dtype=np.float64),
        'Bd':       np.ascontiguousarray(Bd, dtype=np.float64),
        'alpha':    float(alpha),
        'h_coeffs': np.ascontiguousarray(h_coeffs, dtype=np.float64),
        'p_coeffs': np.ascontiguousarray(p_coeffs, dtype=np.float64),
        'Vinv':     np.ascontiguousarray(Vinv.real.flatten(), dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Grid-search entry point — scalar ISE, no time-series
# ---------------------------------------------------------------------------

def simulate_closed_loop_analytical(
    patient,
    double kp, double ki, double kd,
    double[:] Ad_arr,
    double[:] Bd_arr,
    double alpha,
    double[:] h_coeffs_arr,
    double[:] p_coeffs_arr,
    double[:] Vinv_arr,
    double duration=60.0,
    double dt=0.1,
    double bis_target=50.0,
    double t_induction=2.0,
    double t_maintenance=30.0,
):
    from physio.schnider import params_from_patient

    cdef double x1, x2, x3, Ce
    cdef double ox1, ox2, ox3
    cdef double e0, emax, ec50, gamma
    cdef double u, bis, t, ise
    cdef int    n_steps, i
    cdef PIDState pid

    cdef double *Ad       = &Ad_arr[0]
    cdef double *Bd       = &Bd_arr[0]
    cdef double *h_coeffs = &h_coeffs_arr[0]
    cdef double *p_coeffs = &p_coeffs_arr[0]
    cdef double *Vinv     = &Vinv_arr[0]

    params = params_from_patient(patient)
    e0     = params[9];  emax = params[10]; ec50 = params[11]; gamma = params[12]

    x1 = 0.0;  x2 = 0.0;  x3 = 0.0;  Ce = 0.0

    pid.kp = kp;  pid.ki = ki;  pid.kd = kd
    pid.setpoint = bis_target;  pid.dt = dt
    pid.max_rate = 300_000.0;   pid.min_rate = 0.0
    pid.integral = 0.0;  pid.prev_measurement = 0.0;  pid.first_step = 1

    t       = 0.0
    ise     = 0.0
    n_steps = int(round(duration / dt))

    for i in range(n_steps):
        bis = bis_fast(Ce, e0, emax, ec50, gamma)
        u   = pid_step(&pid, bis)

        if t >= t_induction and t <= t_maintenance:
            ise += (bis - bis_target) * (bis - bis_target) * dt

        Ce = ce_step(x1, x2, x3, Ce, u, alpha, h_coeffs, p_coeffs, Vinv)
        analytical_step(x1, x2, x3, u, Ad, Bd, &ox1, &ox2, &ox3)
        x1 = ox1;  x2 = ox2;  x3 = ox3
        t += dt

    return ise
