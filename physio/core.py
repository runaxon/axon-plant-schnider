"""
physio.core
===========

Shared infrastructure for compartmental physiological models.

Provides:
    - Patient           : standard demographic covariates
    - rk4_step          : generic fixed-step RK4 integrator
    - simulate          : generic simulation loop
    - SimulationResult  : time-series container

Design notes
------------
Everything here is written as plain functions and dataclasses — no abstract
base classes, no virtual dispatch.  Each model module (schnider, hovorka, …)
must expose two callables that plug into simulate():

    derivatives(state, inputs, params) -> tuple[float, ...]
    outputs(state, params)             -> dict[str, float]

The state is always a plain tuple of floats so that the hot path
(derivatives + rk4_step) contains only arithmetic — a requirement for clean
transpilation to C / Cython / Numba without object overhead.

Time is tracked by the simulation loop, not stored in the state tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Patient covariates
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    """Standard demographic covariates used across compartmental PK/PD models."""
    age: float  # years
    weight: float  # kg  (total body weight)
    height: float  # cm
    sex: str  # 'male' or 'female'

    def lean_body_mass(self) -> float:
        """From Schnider TW et al. 1998"""
        if self.sex.lower() == 'male':
            return 1.1 * self.weight - 128 * (self.weight / self.height) ** 2
        else:
            return 1.07 * self.weight - 148 * (self.weight / self.height) ** 2

    def bmi(self) -> float:
        return self.weight / (self.height / 100) ** 2


# ---------------------------------------------------------------------------
# Generic RK4 integrator  (state = plain tuple of floats, time separate)
# ---------------------------------------------------------------------------

DerivativesFn = Callable[[tuple, tuple, tuple], tuple]
"""
Signature: deriv(state: tuple, inputs: tuple, params: tuple) -> tuple

- state  : current compartment amounts / concentrations
- inputs : exogenous inputs held constant over the step (e.g. infusion rates)
- params : model parameter tuple — never mutated
Returns a tuple of derivatives, one per state element.
"""

def rk4_step(
    deriv: DerivativesFn,
    state: tuple,
    inputs: tuple,
    params: tuple,
    dt: float,
) -> tuple:
    """
    Advance `state` by one fixed RK4 step of size `dt`.

    Time is not part of the state — the caller is responsible for tracking it.
    `inputs` and `params` are held constant over the step.
    """
    k1 = deriv(state, inputs, params)

    s2 = tuple(s + 0.5 * dt * d for s, d in zip(state, k1))
    k2 = deriv(s2, inputs, params)

    s3 = tuple(s + 0.5 * dt * d for s, d in zip(state, k2))
    k3 = deriv(s3, inputs, params)

    s4 = tuple(s + dt * d for s, d in zip(state, k3))
    k4 = deriv(s4, inputs, params)

    return tuple(
        s + (dt / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
        for s, d1, d2, d3, d4 in zip(state, k1, k2, k3, k4)
    )

# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """
    Time-series output from simulate().

    `time` is always populated.  `outputs` is a dict of named signal arrays
    whose keys are defined by each model's outputs() function, e.g.:
        result.outputs['cp']   # plasma concentration
        result.cp              # shorthand via __getattr__
    """
    time: list[float] = field(default_factory=list)
    outputs: dict[str, list[float]] = field(default_factory=dict)

    def __getattr__(self, name: str) -> list[float]:
        """Allow result.bis, result.cp, etc. as shorthand."""
        try:
            return self.outputs[name]
        except KeyError:
            raise AttributeError(f"SimulationResult has no output '{name}'") from None

# ---------------------------------------------------------------------------
# Generic simulation loop
# ---------------------------------------------------------------------------
def simulate(
    deriv: DerivativesFn,
    outputs_fn: Callable[[tuple, tuple], dict[str, float]],
    state0: tuple,
    params: tuple,
    infusion_schedule: list[tuple[float, float | tuple]],
    duration: float,
    dt: float=0.1
) -> SimulationResult:
    """
    Run a forward simulation of any compartmental model.

    Parameters
    ----------
    deriv :
        Model ODE right-hand side: deriv(state, inputs, params) -> tuple.
    outputs_fn :
        Computes named observable signals from state:
        outputs_fn(state, params) -> dict[str, float].
    state0 :
        Initial state tuple (all zeros for a drug-naive subject).
    params :
        Model parameter tuple, produced by the model's params_from_patient().
    infusion_schedule :
        List of (time, inputs) pairs.  `inputs` is a float or tuple of floats
        for multi-input models.  Rates are step-wise constant between entries.
    duration :
        Total simulation time (minutes, or model's native time unit).
    dt :
        RK4 step size.  Default 0.1 min (6 s).

    Returns
    -------
    SimulationResult with time-series for every key in outputs_fn.
    """
    schedule = sorted(infusion_schedule, key=lambda x: x[0])

    def current_inputs(t: float) -> tuple:
        inp = 0.0
        for (t_start, r) in schedule:
            if t >= t_start:
                inp = r
            else:
                break
        return inp if isinstance(inp, tuple) else (inp,)

    result = SimulationResult()
    state = state0
    t = 0.0
    n_steps = int(round(duration / dt))

    for _ in range(n_steps):
        inputs = current_inputs(t)

        result.time.append(t)
        for k, v in outputs_fn(state, params).items():
            result.outputs.setdefault(k, []).append(v)

        state = rk4_step(deriv, state, inputs, params, dt)
        t += dt

    # Append the final point after the last step
    result.time.append(t)
    for k, v in outputs_fn(state, params).items():
        result.outputs.setdefault(k, []).append(v)

    return result
