"""
physio.schnider
===============

Schnider (1998/1999) propofol PK/PD model.

Pharmacokinetics  — Schnider TW et al. Anesthesiology 88(5):1170-82, 1998
Pharmacodynamics  — Schnider TW et al. Anesthesiology 90(6):1502-16, 1999

Model summary
-------------
3-compartment mammillary PK model with an effect-site link compartment.
Plasma concentration drives equilibration to the effect site (Ce) via ke0.
BIS is computed from Ce via a sigmoidal Emax (Hill) equation.

State tuple layout  (used by derivatives / outputs)
--------------------
    index 0 : x1  — drug amount in central compartment      (µg)
    index 1 : x2  — drug amount in fast peripheral           (µg)
    index 2 : x3  — drug amount in slow peripheral           (µg)
    index 3 : xe  — drug amount in effect-site compartment   (µg)

Inputs tuple layout
-------------------
    index 0 : infusion rate  (µg/min)

Params tuple layout  — produced by params_from_patient()
-------------------
    index 0  : v1    central volume          (L)
    index 1  : v2    fast peripheral volume  (L)
    index 2  : v3    slow peripheral volume  (L)
    index 3  : k10   elimination rate        (1/min)
    index 4  : k12   (1/min)
    index 5  : k13   (1/min)
    index 6  : k21   (1/min)
    index 7  : k31   (1/min)
    index 8  : ke0   effect-site equil. rate (1/min)
    index 9  : e0    baseline BIS
    index 10 : emax  maximum BIS reduction
    index 11 : ec50  Ce at half-max effect   (µg/mL)
    index 12 : gamma Hill coefficient
"""

from __future__ import annotations

from physio.core import Patient, SimulationResult, simulate as _simulate

# ---------------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------------
def params_from_patient(patient: Patient) -> tuple:
    """
    Compute the Schnider population PK/PD parameter tuple from patient covariates.

    PK parameters are covariate-adjusted (Schnider 1998).
    PD parameters are population mean values (Schnider 1999).
    """
    lbm = patient.lean_body_mass()

    # --- PK (Schnider 1998) ---
    v1  = 4.27
    v2  = 18.9 - 0.391 * (patient.age - 53)
    v3  = 238.0

    cl1 = (
        1.89
        + 0.0456 * (patient.weight - 77)
        - 0.0681 * (lbm - 59)
        + 0.0264 * (patient.height - 177)
    )
    cl2 = 1.29 - 0.024 * (patient.age - 53)
    cl3 = 0.836

    k10 = cl1 / v1
    k12 = cl2 / v1
    k13 = cl3 / v1
    k21 = cl2 / v2
    k31 = cl3 / v3

    # --- PD (Schnider 1999) ---
    ke0   = 0.456
    e0    = 97.4
    emax  = 97.4
    ec50  = 3.08
    gamma = 1.47

    return (v1, v2, v3, k10, k12, k13, k21, k31, ke0, e0, emax, ec50, gamma)


# Initial state: drug-naive patient (all compartments empty)
STATE0: tuple = (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Derivatives  —  the hot path, transpilation target
# ---------------------------------------------------------------------------
def derivatives(state: tuple, inputs: tuple, params: tuple) -> tuple:
    """
    Schnider ODE right-hand side.

    Returns (dx1/dt, dx2/dt, dx3/dt, dxe/dt) in µg/min.

    This is the function that will be profiled and transpiled to C.
    It contains only arithmetic on plain floats — no Python objects.
    """
    x1, x2, x3, xe = state
    (u,) = inputs   # infusion rate (µg/min)
    v1, v2, v3, k10, k12, k13, k21, k31, ke0 = params[:9]

    dx1 = u - (k10 + k12 + k13) * x1 + k21 * x2 + k31 * x3
    dx2 = k12 * x1 - k21 * x2
    dx3 = k13 * x1 - k31 * x3
    dxe = ke0 * (x1 - xe)   # amounts track together; Ce = xe / (v1 * 1000) µg/mL

    return (dx1, dx2, dx3, dxe)


# ---------------------------------------------------------------------------
# Outputs  —  computes observable signals from the current state
# ---------------------------------------------------------------------------

def outputs(state: tuple, params: tuple) -> dict[str, float]:
    """
    Compute named observable signals from the current state.

    Returns
    -------
    dict with keys: 'cp', 'ce', 'bis'
        cp   : plasma concentration          (µg/mL)
        ce   : effect-site concentration     (µg/mL)
        bis  : BIS score  (0–100)
    """
    x1, x2, x3, xe = state
    v1, v2, v3, k10, k12, k13, k21, k31, ke0, e0, emax, ec50, gamma = params

    cp = x1 / (v1 * 1000.0)   # µg / (L * 1000 mL/L) = µg/mL
    ce = xe / (v1 * 1000.0)

    if ce <= 0.0:
        bis = e0
    else:
        bis = max(0.0, e0 - emax * (ce ** gamma) / (ec50 ** gamma + ce ** gamma))

    return {'cp': cp, 'ce': ce, 'bis': bis}


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def simulate(
    patient:           Patient,
    infusion_schedule: list[tuple[float, float]],
    duration:          float,
    dt:                float = 0.1,
    params:            tuple | None = None,
) -> SimulationResult:
    """
    Simulate the Schnider model for a given patient and infusion schedule.

    Parameters
    ----------
    patient :
        Patient demographics.
    infusion_schedule :
        List of (time_min, rate_mcg_per_min) pairs.  Step-wise constant.
    duration :
        Total simulation time (minutes).
    dt :
        RK4 step size (minutes).  Default 0.1 min = 6 s.
    params :
        Override population parameters.  If None, computed from patient.

    Returns
    -------
    SimulationResult  with outputs: 'cp', 'ce', 'bis'
    """
    if params is None:
        params = params_from_patient(patient)

    return _simulate(
        deriv      = derivatives,
        outputs_fn = outputs,
        state0     = STATE0,
        params     = params,
        infusion_schedule = infusion_schedule,
        duration   = duration,
        dt         = dt,
    )
