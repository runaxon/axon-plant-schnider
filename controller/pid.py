"""
controller.pid
==============

Discrete-time PID controller for closed-loop anesthesia infusion.

The controller operates on BIS error and outputs an infusion rate (µg/min).
It is model-agnostic — it sees only the BIS signal and knows nothing about
the underlying PK/PD model.

Design notes
------------
- Anti-windup via integrator clamping: the integral term is not accumulated
  when the output is saturated. This prevents the integrator from winding up
  during induction when the output is pinned at the maximum rate.
- Output clamping: infusion rate is bounded to [0, max_rate]. Negative rates
  are meaningless (you cannot extract drug from a patient).
- Derivative on measurement: the derivative term acts on the BIS measurement
  directly rather than the error, avoiding derivative kick when the setpoint
  changes.
- Stateless step interface: PIDController.step() takes the current BIS and
  returns the new infusion rate. All state (integral, previous measurement)
  is stored on the instance.
"""

from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class PIDController:
    """
    Discrete-time PID controller.

    Parameters
    ----------
    kp       : Proportional gain
    ki       : Integral gain
    kd       : Derivative gain
    setpoint : BIS target (default 50)
    dt       : Control loop time step (minutes, must match simulation dt)
    max_rate : Maximum infusion rate (µg/min)
    min_rate : Minimum infusion rate (µg/min), default 0
    """
    kp:       float
    ki:       float
    kd:       float
    setpoint: float = 50.0
    dt:       float = 0.1          # minutes
    max_rate: float = 300_000.0    # µg/min (~300 mg/min, well above any clinical dose)
    min_rate: float = 0.0

    # Internal state — reset between patients
    _integral:    float = field(default=0.0, init=False, repr=False)
    _prev_measurement: float = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Reset controller state. Call between patients."""
        self._integral         = 0.0
        self._prev_measurement = None

    def step(self, measurement: float) -> float:
        """
        Compute the next infusion rate given the current BIS measurement.

        Parameters
        ----------
        measurement : current BIS reading

        Returns
        -------
        Infusion rate in µg/min, clamped to [min_rate, max_rate].
        """
        # BIS is an inverse response: higher infusion → lower BIS.
        # Error is defined as measurement - setpoint so that positive error
        # (BIS too high) produces positive output (increase infusion rate).
        error = measurement - self.setpoint

        # Proportional
        p = self.kp * error

        # Derivative on measurement (avoids kick on setpoint change).
        # Positive when BIS is rising (increase infusion),
        # negative when BIS is falling (reduce infusion to avoid overshoot).
        if self._prev_measurement is None:
            d = 0.0
        else:
            d = self.kd * (measurement - self._prev_measurement) / self.dt

        # Integral term using accumulated sum
        i = self.ki * self._integral

        # Tentative output
        output = p + i + d

        # Anti-windup: only accumulate integral when output is not saturated
        saturated = output >= self.max_rate or output <= self.min_rate
        if not saturated:
            self._integral += error * self.dt

        self._prev_measurement = measurement

        return max(self.min_rate, min(self.max_rate, output))
