"""
Demonstrates the buggy effect-site ODE: dxe_dt = ke0 * (x1 - xe)
where xe is stored as an amount (µg) and outputs converts via xe / (v1 * 1000).

This implies Ve = V1, which is wrong. Ce equilibrates near-instantly
because the driving force (x1 - xe) is in µg (~thousands) instead of
µg/mL (~single digits), making ke0 effectively ~4270x larger.

Compare to demo_schnider.py which uses the correct dCe_dt = ke0 * (C1 - Ce).
"""

import matplotlib.pyplot as plt

from physio.core import Patient, rk4_step, simulate as _simulate, SimulationResult
from physio.schnider import params_from_patient
from plot_style import apply as apply_style, COLORS


def derivatives_buggy(state, inputs, params):
    """Original buggy ODE: mixes amounts and concentrations in effect-site."""
    (u_t,) = inputs
    x1, x2, x3, xe = state
    v1, v2, v3, k10, k12, k13, k21, k31, ke0 = params[:9]

    dx1_dt = u_t - (k10 + k12 + k13) * x1 + k21 * x2 + k31 * x3
    dx2_dt = k12 * x1 - k21 * x2
    dx3_dt = k13 * x1 - k31 * x3
    dxe_dt = ke0 * (x1 - xe)   # BUG: x1 is µg, xe is µg — implies Ve = V1

    return (dx1_dt, dx2_dt, dx3_dt, dxe_dt)


def outputs_buggy(state, params):
    """Convert xe (amount) to Ce (concentration) via xe / (v1 * 1000)."""
    x1, x2, x3, xe = state
    v1, v2, v3, k10, k12, k13, k21, k31, ke0, e0, emax, ec50, gamma = params

    cp = x1 / (v1 * 1000.0)
    ce = xe / (v1 * 1000.0)   # unit conversion — correct form, wrong xe

    if ce <= 0.0:
        bis = e0
    else:
        bis = max(0.0, e0 - emax * (ce ** gamma) / (ec50 ** gamma + ce ** gamma))

    return {'cp': cp, 'ce': ce, 'bis': bis}


patient = Patient(age=40, weight=70, height=170, sex='male')
params = params_from_patient(patient)

schedule = [
    (0.0, 140_000.0),
    (1.0,   7_000.0),
    (30.0,      0.0),
]

result = _simulate(
    deriv=derivatives_buggy,
    outputs_fn=outputs_buggy,
    state0=(0.0, 0.0, 0.0, 0.0),
    params=params,
    infusion_schedule=schedule,
    duration=90.0,
    dt=0.05,
)

print(f"Peak Cp  : {max(result.cp):.2f} µg/mL")
print(f"Peak Ce  : {max(result.ce):.2f} µg/mL")
print(f"Nadir BIS: {min(result.bis):.1f}")

apply_style()
C = COLORS

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.subplots_adjust(hspace=0.08)

axes[0].plot(result.time, result.cp, label='Cp (plasma)', color=C['teal'], linewidth=1.5)
axes[0].plot(result.time, result.ce, label='Ce (effect-site, BUGGY)', color=C['red'], linestyle='--', linewidth=1.5)
axes[0].set_ylabel('Concentration (µg/mL)', fontsize=9)
axes[0].legend(fontsize=8)
axes[0].set_title('Schnider PK/PD — BUGGY effect-site ODE', color=C['text'], fontsize=11, pad=10)

axes[1].plot(result.time, result.bis, color=C['green'], linewidth=1.5)
axes[1].axhline(60, color=C['red'], linestyle=':', linewidth=1, label='BIS 60 target')
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
axes[2].step(result.time, infusion_mg, where='post', color=C['purple'], linewidth=1.5)
axes[2].set_ylabel('Infusion (mg/min)', fontsize=9)
axes[2].set_xlabel('Time (min)', fontsize=9)

plt.savefig('schnider_demo_buggy.png', dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved to schnider_demo_buggy.png')
