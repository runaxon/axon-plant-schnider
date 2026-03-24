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

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(result.time, result.cp, label='Cp (plasma)', color='steelblue')
axes[0].plot(result.time, result.ce, label='Ce (effect-site)', color='darkorange', linestyle='--')
axes[0].set_ylabel('Concentration (µg/mL)')
axes[0].legend()
axes[0].set_title('Schnider Propofol PK/PD')

axes[1].plot(result.time, result.bis, color='green')
axes[1].axhline(60, color='red', linestyle=':', linewidth=1, label='BIS 60 target')
axes[1].set_ylabel('BIS')
axes[1].set_ylim(0, 100)
axes[1].legend()

infusion_mg = []
for t in result.time:
    rate = 0.0
    for (t_start, r) in schedule:
        if t >= t_start:
            rate = r
    infusion_mg.append(rate / 1000.0)
axes[2].step(result.time, infusion_mg, where='post', color='purple')
axes[2].set_ylabel('Infusion (mg/min)')
axes[2].set_xlabel('Time (min)')

plt.tight_layout()
plt.savefig('schnider_demo.png', dpi=150)
plt.show()
print('Plot saved to schnider_demo.png')
