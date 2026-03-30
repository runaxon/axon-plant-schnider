"""
physio.schnider
===============

Schnider (1998/1999) propofol PK/PD model.

Pharmacokinetics — Schnider TW et al. Anesthesiology 88(5):1170-82, 1998
Pharmacodynamics — Schnider TW et al. Anesthesiology 90(6):1502-16, 1999

Model summary
-------------
3-compartment mammillary PK model with an effect-site link compartment.
Plasma concentration drives equilibration to the effect site (Ce) via ke0.
BIS is computed from Ce via a sigmoidal Emax (Hill) equation.

State tuple layout  (used by derivatives / outputs)
--------------------
    index 0 : x1 — drug amount in central compartment       (µg)
    index 1 : x2 — drug amount in fast peripheral           (µg)
    index 2 : x3 — drug amount in slow peripheral           (µg)
    index 3 : Ce — effect-site concentration                (µg/mL)

    Note: Ce is stored as concentration, not amount. Ve is negligible (Hull) and
    k1e is negligible relative to ke0 (Sheiner), so the effect-site decouples from
    the central compartment and the ODE reduces to dCe/dt = ke0 * (C1 - Ce).
    Ve and k1e cancel exactly and do not appear anywhere in the model.

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
    v1 = 4.27
    v2 = 18.9 - 0.391 * (patient.age - 53)
    v3 = 238.0

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
    ke0 = 0.456
    e0 = 97.4
    emax = 97.4
    ec50 = 3.08
    gamma = 1.47

    return (v1, v2, v3, k10, k12, k13, k21, k31, ke0, e0, emax, ec50, gamma)

# Initial state: drug-naive patient (all compartments empty)
# C1, C2, C3, and Sink
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
    # Infusion rate (µg/min)
    (u_t, ) = inputs  

    # x1, x2, x3 are drug amounts (µg); Ce is effect-site concentration (µg/mL)
    x1, x2, x3, Ce = state

    # Volumes and rate parameters
    v1, v2, v3, k10, k12, k13, k21, k31, ke0 = params[:9]

    # Central compartment plasma concentration (µg/mL)
    # v1 is in litres; 1 L = 1000 mL, so µg/L = µg/(1000 mL) → divide by 1000
    C1 = x1 / (v1 * 1000.0)

    # Amounts ODEs (µg/min)
    dx1_dt = u_t - (k10 + k12 + k13) * x1 + k21 * x2 + k31 * x3
    dx2_dt = k12 * x1 - k21 * x2
    dx3_dt = k13 * x1 - k31 * x3

    # Effect-site concentration ODE (µg/mL/min)
    # dCe/dt = ke0 * (C1 - Ce) — exact, Ve and k1e cancel (Sheiner + Hull)
    dCe_dt = ke0 * (C1 - Ce)

    return (dx1_dt, dx2_dt, dx3_dt, dCe_dt)

def _run_mass_balance(params, infusion_rate, duration, dt, buggy):
    """
    5-state Euler integration for mass balance verification.
    State: (x1, x2, x3, A_eliminated, Ce)
      x1, x2, x3   — drug amounts in compartments 1/2/3  (µg)
      A_eliminated — cumulative drug eliminated via k10  (µg)
      Ce           — effect-site concentration           (µg/mL)

    Ce has negligible mass (Hull) and is excluded from the amount balance.

    buggy=True reproduces the original error: dCe_dt = ke0 * (x1 - Ce),
    which mixes amounts and concentrations, implying Ve = V1.
    buggy=False uses the correct form: dCe_dt = ke0 * (C1 - Ce).
    """
    v1, v2, v3, k10, k12, k13, k21, k31, ke0 = params[:9]

    x1, x2, x3, A_elim, Ce = 0.0, 0.0, 0.0, 0.0, 0.0
    t = 0.0

    while t < duration:
        C1 = x1 / (v1 * 1000.0)   # µg/mL

        dx1 = infusion_rate - (k10 + k12 + k13) * x1 + k21 * x2 + k31 * x3
        dx2 = k12 * x1 - k21 * x2
        dx3 = k13 * x1 - k31 * x3
        delim = k10 * x1

        # BUG: mixes amounts and concentrations — implies Ve = V1
        # dCe = ke0 * (x1 - Ce)
        #
        # FIX: both sides are concentrations (µg/mL), Ve and k1e cancel exactly
        if buggy:
            dCe = ke0 * (x1 - Ce)   # wrong: x1 is µg, Ce is µg/mL
        else:
            dCe = ke0 * (C1 - Ce)   # correct: C1 = x1 / (v1 * 1000)

        x1 += dx1   * dt
        x2 += dx2   * dt
        x3 += dx3   * dt
        A_elim += delim * dt
        Ce += dCe   * dt

        t += dt

    total_infused = infusion_rate * duration
    total_in_system = x1 + x2 + x3 + A_elim
    mass_balance_error = total_in_system - total_infused

    return {
        "total_infused_ug":   total_infused,
        "x1_ug":              x1,
        "x2_ug":              x2,
        "x3_ug":              x3,
        "A_eliminated_ug":    A_elim,
        "total_in_system_ug": total_in_system,
        "mass_balance_error": mass_balance_error,
        "Ce_ugml":            Ce,
    }


def run_mass_balance_test(params, infusion_rate, duration=60, dt=0.01):
    """
    Run the mass balance check with both the buggy and fixed effect-site ODE.
    Prints a comparison report and returns (buggy_result, fixed_result).
    """
    buggy = _run_mass_balance(params, infusion_rate, duration, dt, buggy=True)
    fixed = _run_mass_balance(params, infusion_rate, duration, dt, buggy=False)

    total = fixed["total_infused_ug"]

    print("=" * 55)
    print(f"  Mass balance test  ({duration} min, {infusion_rate:.0f} µg/min)")
    print("=" * 55)
    print(f"  {'':30s} {'BUGGY':>10}  {'FIXED':>10}")
    print(f"  {'x1 (µg)':30s} {buggy['x1_ug']:>10.1f}  {fixed['x1_ug']:>10.1f}")
    print(f"  {'x2 (µg)':30s} {buggy['x2_ug']:>10.1f}  {fixed['x2_ug']:>10.1f}")
    print(f"  {'x3 (µg)':30s} {buggy['x3_ug']:>10.1f}  {fixed['x3_ug']:>10.1f}")
    print(f"  {'A_eliminated (µg)':30s} {buggy['A_eliminated_ug']:>10.1f}  {fixed['A_eliminated_ug']:>10.1f}")
    print(f"  {'xe (µg) / Ce (µg/mL)':30s} {buggy['Ce_ugml']:>10.4f}  {fixed['Ce_ugml']:>10.4f}")
    print(f"  {'-'*53}")
    print(f"  {'total in system (µg)':30s} {buggy['total_in_system_ug']:>10.1f}  {fixed['total_in_system_ug']:>10.1f}")
    print(f"  {'total infused (µg)':30s} {total:>10.1f}  {total:>10.1f}")
    print(f"  {'error (µg)':30s} {buggy['mass_balance_error']:>10.1f}  {fixed['mass_balance_error']:>10.4f}")
    print(f"  {'error (%)':30s} {100*buggy['mass_balance_error']/total:>10.4f}  {100*fixed['mass_balance_error']/total:>10.6f}")
    print("=" * 55)

    return buggy, fixed


# ---------------------------------------------------------------------------
# Outputs — computes observable signals from the current state
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
    x1, _, _, Ce = state
    v1, v2, v3, k10, k12, k13, k21, k31, ke0, e0, emax, ec50, gamma = params

    cp = x1 / (v1 * 1000.0)   # µg / (L * 1000 mL/L) = µg/mL
    ce = Ce                    # already in µg/mL

    if ce <= 0.0:
        bis = e0
    else:
        bis = max(0.0, e0 - emax * (ce ** gamma) / (ec50 ** gamma + ce ** gamma))

    return { 'cp': cp, 'ce': ce, 'bis': bis }

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def simulate(
    patient: Patient,
    infusion_schedule: list[tuple[float, float]],
    duration: float,
    dt: float = 0.1,
    params: tuple | None = None,
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
        deriv=derivatives,
        outputs_fn=outputs,
        state0=STATE0,
        params=params,
        infusion_schedule=infusion_schedule,
        duration=duration,
        dt=dt
    )
