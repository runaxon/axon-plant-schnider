import matplotlib.pyplot as plt

from physio.core import Patient
from physio.schnider import params_from_patient, run_mass_balance_test, _run_mass_balance
from plot_style import apply as apply_style, COLORS

patient = Patient(sex='male', weight=70, height=175, age=40)
params = params_from_patient(patient)

# --- Comparison table at dt=0.001 ---
run_mass_balance_test(params, infusion_rate=8000.0, duration=60, dt=0.001)

# --- dt sweep ---
# Use a duration that does not divide evenly by any dt to avoid
# floating-point cancellation masking the true integration error.
SWEEP_DURATION = 60.3
dts = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
errors = []
for dt in dts:
    r = _run_mass_balance(params, 8000.0, SWEEP_DURATION, dt, buggy=False)
    pct = abs(100 * r['mass_balance_error'] / r['total_infused_ug'])
    errors.append(pct)
    print(f"dt={dt:.4f}  error={r['mass_balance_error']:+.4f} µg  ({pct:.6f}%)")

apply_style()
C = COLORS

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(dts[::-1], errors[::-1], color=C['teal'], marker='o', linewidth=1.5, markersize=5)

# Reference O(dt) slope line through the coarsest point
ref_x = [dts[-1], dts[0]]
ref_y = [errors[-1], errors[-1] * (dts[0] / dts[-1])]
ax.loglog(ref_x[::-1], ref_y[::-1], color=C['orange'], linestyle='--', linewidth=1, label='O(dt) reference')

ax.set_xlabel('dt (min)', fontsize=9)
ax.set_ylabel('Mass balance error (%)', fontsize=9)
ax.set_title('Euler integration error vs step size', fontsize=10, pad=8)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('mass_balance_error.png', dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved to mass_balance_error.png')
