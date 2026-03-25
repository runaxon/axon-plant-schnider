"""
Cohort evaluation harness for the Schnider PK/PD model.

Runs a fixed open-loop infusion schedule across a virtual patient cohort and
plots the population spread of BIS and Ce over time. This is the evaluation
function that will be called on every candidate PID controller during
optimization. The controller replaces the fixed schedule.

Usage
-----
    # Generate and save a cohort (do once):
    python eval_cohort.py --generate --n 100 --seed 42 --out cohorts/n100_seed42.json

    # Evaluate against a saved cohort:
    python eval_cohort.py --cohort cohorts/n100_seed42.json
"""

import argparse
import os

import matplotlib.pyplot as plt

from physio.cohort import generate_cohort, cohort_to_json, cohort_from_json
from physio.schnider import simulate
from plot_style import apply as apply_style, COLORS

# ---------------------------------------------------------------------------
# Infusion schedule (open-loop — will be replaced by PID controller output)
# ---------------------------------------------------------------------------
SCHEDULE = [
    (0.0,  140_000.0),   # induction: ~2 mg/kg over 1 min
    (1.0,    7_000.0),   # maintenance: ~100 µg/kg/min
    (30.0,       0.0)    # stop — washout
]

DURATION = 60.0   # minutes
DT       = 0.1    # minutes

# ---------------------------------------------------------------------------
# Loss function parameters
# ---------------------------------------------------------------------------
BIS_TARGET      = 50.0    # maintenance BIS target
T_INDUCTION     = 2.0     # minutes — BIS must cross below 60 by this time
T_MAINTENANCE   = 30.0    # minutes — end of maintenance window (washout excluded)
LAMBDA          = 500_000.0  # induction penalty weight (pre-normalization)

# Worst-case ISE reference: flat-line at BIS=0 over the full maintenance window.
# Used to normalize loss to [0, ~1] regardless of window length or dt.
#   ISE_ref = BIS_target^2 * (T_maintenance - T_induction)
ISE_REF = BIS_TARGET ** 2 * (T_MAINTENANCE - T_INDUCTION)   # 70,000 BIS²·min
LAMBDA_NORM = LAMBDA / ISE_REF                               # ~7.14


# ---------------------------------------------------------------------------
# Core evaluation function — this is what the PID optimizer will call
# ---------------------------------------------------------------------------
def evaluate_cohort(cohort, schedule=SCHEDULE, duration=DURATION, dt=DT):
    """
    Simulate every patient in the cohort under the given infusion schedule.

    Parameters
    ----------
    cohort   : list[Patient]
    schedule : infusion schedule — list of (time, rate_µg_per_min) pairs
    duration : total simulation time (min)
    dt       : RK4 step size (min)

    Returns
    -------
    list[SimulationResult], one per patient in cohort order.
    """
    return [simulate(patient, schedule, duration=duration, dt=dt) for patient in cohort]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------
def compute_loss(results, dt=DT):
    """
    Compute the mean cohort loss for a set of simulation results.

    Loss = (1/N) * sum_n [ ISE_n + induction_penalty_n ]

    ISE is computed over the maintenance window only (T_INDUCTION to T_MAINTENANCE).
    The induction penalty fires if BIS has not crossed below 60 by T_INDUCTION.

    Parameters
    ----------
    results : list[SimulationResult] — one per patient
    dt      : step size used in the simulation (min), used for ISE integration

    Returns
    -------
    float — mean normalized loss across the cohort.
        ISE component is in [0, 1] where 1 = worst-case (BIS=0 throughout).
        Induction penalty adds ~7.14 per failed patient.
    """
    total = 0.0
    for r in results:
        ise     = 0.0
        induced = False

        for t, bis in zip(r.time, r.bis):
            # Check induction criterion
            if t <= T_INDUCTION and bis < 60.0:
                induced = True

            # Accumulate ISE over maintenance window only
            if T_INDUCTION <= t <= T_MAINTENANCE:
                ise += (bis - BIS_TARGET) ** 2 * dt

        induction_penalty = 0.0 if induced else LAMBDA_NORM
        total += ise / ISE_REF + induction_penalty

    return total / len(results)


# ---------------------------------------------------------------------------
# Summary statistics across the cohort at each time point
# ---------------------------------------------------------------------------
def _percentile(values, p):
    """Compute the p-th percentile of a list (linear interpolation)."""
    sorted_v = sorted(values)
    n = len(sorted_v)
    idx = p / 100.0 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo])


def cohort_stats(results, key):
    """
    Compute median, 10th, and 90th percentile time-series for a named output.

    Returns (median, p10, p90) — each a list of floats aligned to results[0].time.
    """
    n_t = len(results[0].time)
    median, p10, p90 = [], [], []
    for i in range(n_t):
        vals = [r.outputs[key][i] for r in results]
        median.append(_percentile(vals, 50))
        p10.append(_percentile(vals, 10))
        p90.append(_percentile(vals, 90))
    return median, p10, p90


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_cohort(results, out_path='cohort_eval.png'):
    apply_style()
    C    = COLORS
    time = results[0].time

    bis_med, bis_p10, bis_p90 = cohort_stats(results, 'bis')
    ce_med,  ce_p10,  ce_p90  = cohort_stats(results, 'ce')

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    # --- BIS ---
    for r in results:
        axes[0].plot(time, r.bis, color=C['green'], linewidth=0.4, alpha=0.2)
    axes[0].plot(time, bis_med, color=C['green'],  linewidth=1.8, label='Median BIS')
    axes[0].fill_between(time, bis_p10, bis_p90, color=C['green'], alpha=0.15, label='P10–P90')
    axes[0].axhline(60, color=C['red'],    linestyle=':', linewidth=1, label='BIS 60 (upper target)')
    axes[0].axhline(40, color=C['orange'], linestyle=':', linewidth=1, label='BIS 40 (lower target)')
    axes[0].set_ylabel('BIS', fontsize=9)
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'Cohort evaluation  (n={len(results)})', color=C['text'], fontsize=11, pad=10)

    # --- Ce ---
    for r in results:
        axes[1].plot(time, r.ce, color=C['teal'], linewidth=0.4, alpha=0.2)
    axes[1].plot(time, ce_med, color=C['teal'],  linewidth=1.8, label='Median Ce')
    axes[1].fill_between(time, ce_p10, ce_p90, color=C['teal'], alpha=0.15, label='P10–P90')
    axes[1].set_ylabel('Ce (µg/mL)', fontsize=9)
    axes[1].set_xlabel('Time (min)', fontsize=9)
    axes[1].legend(fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Plot saved to {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cohort evaluation harness.')
    parser.add_argument('--cohort',   type=str, help='Path to a saved cohort JSON file.')
    parser.add_argument('--generate', action='store_true', help='Generate a new cohort.')
    parser.add_argument('--n',        type=int, default=100, help='Cohort size (with --generate).')
    parser.add_argument('--seed',     type=int, default=42,  help='Random seed (with --generate).')
    parser.add_argument('--out',      type=str, default='cohorts/n100_seed42.json',
                        help='Save path for generated cohort.')
    args = parser.parse_args()

    if args.generate:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        cohort = generate_cohort(n=args.n, seed=args.seed)
        cohort_to_json(cohort, args.out)
        print(f'Saved {args.n}-patient cohort (seed={args.seed}) to {args.out}')
    elif args.cohort:
        cohort = cohort_from_json(args.cohort)
        print(f'Loaded {len(cohort)}-patient cohort from {args.cohort}')
    else:
        # Default: generate in-memory for a quick run
        cohort = generate_cohort(n=100, seed=42)
        print(f'Using in-memory cohort (n=100, seed=42)')

    results = evaluate_cohort(cohort)

    bis_med, bis_p10, bis_p90 = cohort_stats(results, 'bis')
    ce_med,  _,       _       = cohort_stats(results, 'ce')
    loss = compute_loss(results)

    print(f'Median nadir BIS : {min(bis_med):.1f}')
    print(f'P10   nadir BIS  : {min(bis_p10):.1f}')
    print(f'P90   nadir BIS  : {min(bis_p90):.1f}')
    print(f'Median peak Ce   : {max(ce_med):.2f} µg/mL')
    print(f'Cohort loss      : {loss:,.1f}')

    plot_cohort(results)
