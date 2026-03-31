"""
Closed-form (analytical) solution for the Schnider 3-compartment PK model.

For piecewise-constant infusion the system dx/dt = Ax + Bu is linear and
time-invariant over each interval. The exact solution is:

    x(t+dt) = Ad * x(t) + Bd * u

where Ad = expm(A*dt) and Bd = A^-1 * (Ad - I) * B are precomputed once
per (patient, dt) pair. Each simulation step is then just a 3x3 matrix-vector
multiply plus a vector-scalar multiply — no per-step matrix ops.

Ce decouples from the 3-compartment system (negligible mass assumption).
With C1(t) = scale * x1(t) as a sum of exponential modes, the Ce update also
has an exact closed form precomputed once:

    Ce(t+dt) = alpha * Ce(t) + sum_i [h_coeff_i + p_coeff_i * u]

where alpha = exp(-ke0*dt) and h/p coefficients depend only on (params, dt).

Usage
-----
    from validation.closed_form import AnalyticalSimulator, analytical_solution
"""

import numpy as np
from scipy.linalg import expm

from physio.core import Patient, SimulationResult
from physio.schnider import params_from_patient


class AnalyticalSimulator:
    """
    Precomputes all dt-dependent matrices for a given (patient, dt) pair.
    Calling simulate() is then just matrix-vector multiplies per step.
    """

    def __init__(self, params, dt):
        v1, v2, v3, k10, k12, k13, k21, k31, ke0 = params[:9]

        self.params = params
        self.dt     = dt
        self.v1     = v1
        self.ke0    = ke0
        self.e0     = params[9]
        self.emax   = params[10]
        self.ec50   = params[11]
        self.gamma  = params[12]

        # Rate matrix and input vector
        A = np.array([
            [-(k10 + k12 + k13),  k21,  k31],
            [ k12,               -k21,    0],
            [ k13,                  0, -k31],
        ])
        B = np.array([1.0, 0.0, 0.0])

        # Discrete-time state transition: x[n+1] = Ad*x[n] + Bd*u[n]
        self.Ad = expm(A * dt)
        self.Bd = np.linalg.solve(A, (self.Ad - np.eye(3))) @ B

        # Eigendecomposition for Ce precomputation
        eigenvalues, V = np.linalg.eig(A)
        Vinv = np.linalg.inv(V)
        lam  = eigenvalues.real
        V0   = V[0, :].real    # first row picks out x1
        d    = (Vinv @ B).real # particular solution coefficients
        scale = 1.0 / (v1 * 1000.0)

        alpha = np.exp(-ke0 * dt)

        # Precompute per-mode coefficients for Ce update:
        #   Ce[n+1] = alpha*Ce[n] + sum_i h_i[n]*c_i + sum_i p_i[n]*u[n]
        # where c_i = (Vinv @ x[n])[i] are the modal coordinates of x[n].
        #
        # h_i: homogeneous contribution from mode i of x[n]
        # p_i: particular contribution from mode i (multiplied by u)
        self._lam   = lam
        self._V0    = V0
        self._d     = d
        self._Vinv  = Vinv
        self._alpha = alpha
        self._scale = scale

        # Precompute scalar integrals (depend only on lam, ke0, dt)
        self._I1 = np.zeros(3)   # integral for homogeneous term
        self._I2 = np.zeros(3)   # integral for particular term (part 1)
        self._I3 = (np.exp(ke0 * dt) - 1.0) / ke0  # integral for particular term (part 2)

        for i in range(3):
            li = lam[i]
            if abs(li + ke0) < 1e-12:
                self._I1[i] = dt
                self._I2[i] = dt
            else:
                val = (np.exp((li + ke0) * dt) - 1.0) / (li + ke0)
                self._I1[i] = val
                self._I2[i] = val

    def step(self, x, Ce, u):
        """Advance one step exactly. Returns (x_new, Ce_new)."""
        # Modal coordinates of current x
        c = self._Vinv @ x   # shape (3,)

        Ce_new = self._alpha * Ce

        for i in range(3):
            li     = self._lam[i]
            V0i    = self._V0[i]
            ci     = c[i].real
            di     = self._d[i]

            # Homogeneous contribution
            h_coeff = self._scale * V0i * ci
            Ce_new += self.ke0 * self._alpha * h_coeff * self._I1[i]

            # Particular contribution (scaled by u)
            p_coeff = self._scale * V0i * di / li * u
            Ce_new += self.ke0 * self._alpha * p_coeff * (self._I2[i] - self._I3)

        x_new = self.Ad @ x + self.Bd * u
        return x_new, Ce_new

    def simulate(self, infusion_schedule, duration):
        """Run a full simulation. Returns SimulationResult."""
        schedule = sorted(infusion_schedule, key=lambda s: s[0])

        def current_input(t):
            u = 0.0
            for (t_start, r) in schedule:
                if t >= t_start:
                    u = r
                else:
                    break
            return u

        result  = SimulationResult()
        x       = np.zeros(3)
        Ce      = 0.0
        t       = 0.0
        n_steps = int(round(duration / self.dt))

        for _ in range(n_steps):
            u  = current_input(t)
            C1 = x[0] / (self.v1 * 1000.0)

            ce_g = Ce ** self.gamma if Ce > 0.0 else 0.0
            bis  = max(0.0, self.e0 - self.emax * ce_g / (self.ec50 ** self.gamma + ce_g)) if Ce > 0.0 else self.e0

            result.time.append(t)
            result.outputs.setdefault('cp', []).append(C1)
            result.outputs.setdefault('ce', []).append(Ce)
            result.outputs.setdefault('bis', []).append(bis)

            x, Ce = self.step(x, Ce, u)
            t += self.dt

        # Final point
        C1   = x[0] / (self.v1 * 1000.0)
        ce_g = Ce ** self.gamma if Ce > 0.0 else 0.0
        bis  = max(0.0, self.e0 - self.emax * ce_g / (self.ec50 ** self.gamma + ce_g)) if Ce > 0.0 else self.e0
        result.time.append(t)
        result.outputs['cp'].append(C1)
        result.outputs['ce'].append(Ce)
        result.outputs['bis'].append(bis)

        return result


def analytical_solution(patient, infusion_schedule, duration, dt=0.1, params=None):
    """Convenience wrapper. Returns SimulationResult."""
    if params is None:
        params = params_from_patient(patient)
    sim = AnalyticalSimulator(params, dt)
    return sim.simulate(infusion_schedule, duration)
