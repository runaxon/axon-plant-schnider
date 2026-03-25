"""
Demo: Schnider propofol PK/PD simulation.

Simulates a standard induction + 30-min maintenance + washout for a
reference patient and plots Cp, Ce, and BIS over time.
"""

import matplotlib.pyplot as plt

from physio.core import Patient
from physio.schnider import simulate, params_from_patient


patient = Patient(age=40, weight=70, height=170, sex='male')

schedule = [
    (0.0,  140_000.0),   # induction: 140 mg/min for 1 min (~2 mg/kg)
    (1.0,    7_000.0),   # maintenance: 7 mg/min (~100 µg/kg/min)
    (30.0,       0.0),   # stop — observe washout
]

result = simulate(patient, schedule, duration=60.0, dt=0.05)
pk     = params_from_patient(patient)

print(f"LBM      : {patient.lean_body_mass():.1f} kg")
print(f"V1/V2/V3 : {pk[0]:.2f} / {pk[1]:.2f} / {pk[2]:.2f} L")
print(f"Peak Cp  : {max(result.cp):.2f} µg/mL")
print(f"Peak Ce  : {max(result.ce):.2f} µg/mL")
print(f"Nadir BIS: {min(result.bis):.1f}")

BG      = '#141414'
SURFACE = '#1e1e1e'
BORDER  = '#2a2a2a'
TEXT    = '#d4d4d4'
MUTED   = '#6a6a6a'
TEAL    = '#57c4b8'
ORANGE  = '#d4956a'
GREEN   = '#8ec07c'
RED     = '#cc6666'
PURPLE  = '#b294bb'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    SURFACE,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   MUTED,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'text.color':        TEXT,
    'legend.facecolor':  SURFACE,
    'legend.edgecolor':  BORDER,
    'grid.color':        BORDER,
    'grid.linestyle':    '-',
    'axes.grid':         True,
    'font.family':       'monospace',
})

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.subplots_adjust(hspace=0.08)

axes[0].plot(result.time, result.cp, label='Cp (plasma)', color=TEAL, linewidth=1.5)
axes[0].plot(result.time, result.ce, label='Ce (effect-site)', color=ORANGE, linestyle='--', linewidth=1.5)
axes[0].set_ylabel('Concentration (µg/mL)', fontsize=9)
axes[0].legend(fontsize=8)
axes[0].set_title('Schnider Propofol PK/PD', color=TEXT, fontsize=11, pad=10)

axes[1].plot(result.time, result.bis, color=GREEN, linewidth=1.5)
axes[1].axhline(60, color=RED, linestyle=':', linewidth=1, label='BIS 60 target')
axes[1].set_ylabel('BIS', fontsize=9)
axes[1].set_ylim(0, 100)
axes[1].legend(fontsize=8)

infusion_mg = []
for t in result.time:
    rate = 0.0
    for (t_start, r) in schedule:
        if t >= t_start:
            rate = r
    infusion_mg.append(rate / 1000.0)
axes[2].step(result.time, infusion_mg, where='post', color=PURPLE, linewidth=1.5)
axes[2].set_ylabel('Infusion (mg/min)', fontsize=9)
axes[2].set_xlabel('Time (min)', fontsize=9)

plt.savefig('schnider_demo.png', dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved to schnider_demo.png')
