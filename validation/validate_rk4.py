"""
Validates RK4 integration accuracy against the closed-form analytical solution.

For each dt, runs both the analytical solution and RK4, then computes:
  - Max absolute error in Cp (µg/mL)
  - Max absolute error in Ce (µg/mL)
  - Max absolute error in BIS

Plots error vs dt on a log-log scale to confirm O(dt^4) convergence.

Usage
-----
    python validation/validate_rk4.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from physio.core import Patient
from physio.schnider import simulate, params_from_patient
from validation.closed_form import analytical_solution
from plot_style import apply as apply_style, COLORS

# Reference patient and schedule
patient  = Patient(age=40, weight=70, height=170, sex='male')
params   = params_from_patient(patient)
schedule = [
    (0.0, 140_000.0),
    (1.0,   7_000.0),
    (30.0,      0.0),
]
DURATION = 60.0

# dt values to sweep — from coarse to fine
DTS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

errors_cp  = []
errors_ce  = []
errors_bis = []

for dt in DTS:
    # Compute analytical solution at the exact same dt as RK4 — no interpolation
    ana = analytical_solution(patient, schedule, DURATION, dt=dt, params=params)
    rk4 = simulate(patient, schedule, DURATION, dt=dt, params=params)

    e_cp  = np.max(np.abs(np.array(rk4.cp)  - np.array(ana.cp)))
    e_ce  = np.max(np.abs(np.array(rk4.ce)  - np.array(ana.ce)))
    e_bis = np.max(np.abs(np.array(rk4.bis) - np.array(ana.bis)))

    errors_cp.append(e_cp)
    errors_ce.append(e_ce)
    errors_bis.append(e_bis)

    print(f"dt={dt:.3f}  max|Cp err|={e_cp:.2e} µg/mL  max|Ce err|={e_ce:.2e} µg/mL  max|BIS err|={e_bis:.2e}")

# --- Plot ---
apply_style()
C = COLORS

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.subplots_adjust(wspace=0.35)

pairs = [
    (errors_cp,  'Max |Cp error| (µg/mL)', C['teal']),
    (errors_ce,  'Max |Ce error| (µg/mL)', C['orange']),
    (errors_bis, 'Max |BIS error|',         C['green']),
]

for ax, (errors, ylabel, color) in zip(axes, pairs):
    ax.loglog(DTS, errors, color=color, marker='o', linewidth=1.5, markersize=5)

    # O(dt^4) reference line anchored at the finest point
    ref_y = [errors[-1] * (dt / DTS[-1]) ** 4 for dt in DTS]
    ax.loglog(DTS, ref_y, color='#666666', linestyle='--', linewidth=1, label='O(dt⁴)')

    ax.set_xlabel('dt (min)', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.invert_xaxis()

axes[0].set_title('RK4 vs analytical — integration error', fontsize=10, pad=8)

plt.savefig('validation/rk4_error.png', dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved to validation/rk4_error.png')
