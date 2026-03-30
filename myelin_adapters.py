"""
Myelin adapter wrappers for the Schnider propofol simulator.

These are thin wrappers over the existing physio/ and controller/ modules.
None of the original files are modified — these adapters are the reference
implementation that Myelin targets for optimization.

Usage:
    python main.py --target ../axon-plant-schnider --adapter-mode
"""

import itertools

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '.myelin', 'c_modules'))

from myelin import PlantAdapter, ControllerAdapter, EvaluatorAdapter

from physio.schnider import params_from_patient, outputs, STATE0, derivatives as _schnider_derivatives
from physio.core import Patient, rk4_step as _rk4_step_py
from physio.cohort import generate_cohort
from controller.pid import PIDController

try:
    import SchniderPlant_plant_c as _plant_c
    _USE_SCALAR_KERNEL = hasattr(_plant_c, 'simulate')
    _USE_C_RK4         = hasattr(_plant_c, 'rk4_step')
except ImportError:
    _plant_c           = None
    _USE_SCALAR_KERNEL = False
    _USE_C_RK4         = False


# ---------------------------------------------------------------------------
# Simulation constants (mirror eval_cohort.py to avoid that import chain)
# ---------------------------------------------------------------------------
DURATION      = 60.0    # minutes
DT            = 0.1     # minutes
BIS_TARGET    = 50.0
T_INDUCTION   = 2.0     # minutes
T_MAINTENANCE = 30.0    # minutes
ISE_REF       = BIS_TARGET ** 2 * (T_MAINTENANCE - T_INDUCTION)  # 70,000
LAMBDA_NORM   = 500_000.0 / ISE_REF                              # ~7.14

# Grid definition (mirrors grid_search.py)
N = 8
KP_VALUES = [2 * 4000 / N * x for x in range(1, N + 1)]
KI_VALUES = [2 *  400 / N * x for x in range(1, N + 1)]
KD_VALUES = [2 * 1600 / N * x for x in range(1, N + 1)]


# ---------------------------------------------------------------------------
# PlantAdapter — the Schnider ODE
# ---------------------------------------------------------------------------

class SchniderPlant(PlantAdapter):
    """
    Wraps the Schnider 3-compartment PK/PD ODE for Myelin scalar kernel generation.

    Myelin extracts derivatives() and observe() — both pure arithmetic — and
    generates a single C function covering the full closed-loop simulation.
    """
    state_size  = 4     # (x1, x2, x3, xe)
    input_size  = 1     # (u_t,)  — infusion rate µg/min
    params_size = 13    # (v1, v2, v3, k10, k12, k13, k21, k31, ke0, e0, emax, ec50, gamma)
    target      = 50.0  # BIS setpoint

    def derivatives(self, t: float, state: tuple, inputs: tuple, params: tuple) -> tuple:
        return _schnider_derivatives(state, inputs, params)

    def observe(self, state: tuple, params: tuple) -> float:
        """BIS score — the scalar measurement the controller sees."""
        xe = state[3]
        v1, _, _, _, _, _, _, _, _, e0, emax, ec50, gamma = params
        ce = xe / (v1 * 1000.0)
        if ce <= 0.0:
            return e0
        return max(0.0, e0 - emax * (ce ** gamma) / (ec50 ** gamma + ce ** gamma))


# ---------------------------------------------------------------------------
# ControllerAdapter — discrete-time PID
# ---------------------------------------------------------------------------

class SchniderPIDController(ControllerAdapter):
    """
    Wraps PIDController for Myelin C transpilation.

    Myelin will verify step() is pure arithmetic and generate a C struct
    with integral accumulator and previous-measurement fields.
    """

    def __init__(self, kp: float = 4000.0, ki: float = 400.0, kd: float = 1600.0,
                 setpoint: float = BIS_TARGET, dt: float = DT,
                 max_rate: float = 300_000.0, min_rate: float = 0.0):
        self._pid = PIDController(
            kp=kp, ki=ki, kd=kd,
            setpoint=setpoint, dt=dt,
            max_rate=max_rate, min_rate=min_rate,
        )

    def step(self, measurement: float) -> float:
        return self._pid.step(measurement)

    def reset(self) -> None:
        self._pid.reset()


# ---------------------------------------------------------------------------
# EvaluatorAdapter — PID grid search over a patient cohort
# ---------------------------------------------------------------------------

class SchniderGridEvaluator(EvaluatorAdapter):
    """
    Runs the PID grid search across a synthetic patient cohort.

    Myelin parallelizes evaluate_one() across the parameter grid using
    multiprocessing.Pool with a pool initializer (cohort loaded once per worker).
    """

    def __init__(self, n_patients: int = 200, seed: int = 99):
        self._cohort = generate_cohort(n=n_patients, seed=seed)

    def load_cohort(self):
        return self._cohort

    def param_grid(self) -> list:
        return [
            {'kp': kp, 'ki': ki, 'kd': kd}
            for kp, ki, kd in itertools.product(KP_VALUES, KI_VALUES, KD_VALUES)
        ]

    def run_scenario(self, patient, params: dict) -> float:
        """Run one patient through closed-loop simulation and return scalar loss."""
        pkg_params = params_from_patient(patient)

        # Use scalar C kernel if available — zero Python boundary crossings per step
        if _USE_SCALAR_KERNEL:
            return _plant_c.simulate(
                pkg_params, params['kp'], params['ki'], params['kd'],
                DURATION, DT, T_INDUCTION, T_MAINTENANCE,
            )

        # Python fallback
        controller = PIDController(kp=params['kp'], ki=params['ki'], kd=params['kd'],
                                   setpoint=BIS_TARGET, dt=DT)
        state   = STATE0
        t       = 0.0
        ise     = 0.0
        induced = False

        for _ in range(int(round(DURATION / DT))):
            bis  = SchniderPlant().observe(state, pkg_params)
            rate = controller.step(bis)
            if t <= T_INDUCTION and bis < 60.0:
                induced = True
            if T_INDUCTION <= t <= T_MAINTENANCE:
                ise += (bis - BIS_TARGET) ** 2 * DT
            if _USE_C_RK4:
                state = _plant_c.rk4_step(state, (rate,), pkg_params, DT)
            else:
                state = _rk4_step_py(_schnider_derivatives, state, (rate,), pkg_params, DT)
            t += DT

        induction_penalty = 0.0 if induced else LAMBDA_NORM
        return ise / ISE_REF + induction_penalty

    def evaluate_one(self, params: dict) -> float:
        """Mean loss across the cohort for a single (kp, ki, kd) combination."""
        losses = [self.run_scenario(p, params) for p in self._cohort]
        return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Standalone check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'Scalar kernel: {"yes" if _USE_SCALAR_KERNEL else "no"}, C RK4: {"yes" if _USE_C_RK4 else "no"}')

    plant = SchniderPlant()
    params = params_from_patient(Patient(age=40, weight=70, height=175, sex='male'))
    deriv = plant.derivatives(0.0, STATE0, (7000.0,), params)
    print(f'derivatives OK: {deriv}')

    ctrl = SchniderPIDController()
    rate = ctrl.step(70.0)
    print(f'controller OK: rate={rate:.1f} µg/min')

    print('All adapters OK')