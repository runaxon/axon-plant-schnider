"""
Grid search over PID gains (kp, ki, kd) evaluated on the full patient cohort.

Each candidate controller is run in closed-loop against every patient in the
cohort.  The cohort loss (normalized ISE + induction penalty) is computed for
each grid point.  The best gains and the full results grid are saved to JSON.

This script is intentionally written in pure Python with no parallelism so
that the wall-clock time is representative of the profiling story in the blog.

Usage
-----
    python grid_search.py --cohort cohorts/n200_seed99.json --out results/grid_search.json
"""

import argparse
import json
import math
import os
import time

from physio.cohort import cohort_from_json, generate_cohort
from physio.schnider import simulate, params_from_patient, STATE0, derivatives, outputs
from physio.core import rk4_step
from controller.pid import PIDController
from eval_cohort import compute_loss, DURATION, DT, BIS_TARGET

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
import itertools

N = 8
KP_CENTER = 4000
KI_CENTER = 400
KD_CENTER = 1600
KP_VALUES = [2 * KP_CENTER / N * x for x in range(1, N + 1)]   # proportional gain
KI_VALUES = [2 * KI_CENTER / N * x for x in range(1, N + 1)]   # integral gain
KD_VALUES = [2 * KD_CENTER / N * x for x in range(1, N + 1)]   # derivative gain

# ---------------------------------------------------------------------------
# Closed-loop simulation
# ---------------------------------------------------------------------------

def simulate_closed_loop(patient, controller, duration=DURATION, dt=DT):
    """
    Run a closed-loop simulation of the Schnider model with a PID controller.

    At each time step the controller observes BIS and outputs an infusion rate,
    which is fed into the next RK4 step.

    Parameters
    ----------
    patient    : Patient
    controller : PIDController  (will be reset before use)
    duration   : total simulation time (min)
    dt         : RK4 step size (min)

    Returns
    -------
    SimulationResult-compatible dict with keys: time, cp, ce, bis, rate
    """
    from physio.core import SimulationResult

    params = params_from_patient(patient)
    controller.reset()

    state   = STATE0
    t       = 0.0
    n_steps = int(round(duration / dt))

    result = SimulationResult()

    for _ in range(n_steps):
        out    = outputs(state, params)
        rate   = controller.step(out['bis'])
        inputs = (rate,)

        result.time.append(t)
        for k, v in out.items():
            result.outputs.setdefault(k, []).append(v)
        result.outputs.setdefault('rate', []).append(rate)

        state = rk4_step(derivatives, state, inputs, params, dt)
        t    += dt

    # Final point
    out  = outputs(state, params)
    rate = controller.step(out['bis'])
    result.time.append(t)
    for k, v in out.items():
        result.outputs.setdefault(k, []).append(v)
    result.outputs.setdefault('rate', []).append(rate)

    return result


def evaluate_closed_loop(cohort, controller, dt=DT):
    """Run the controller on every patient and return the cohort loss."""
    results = [simulate_closed_loop(p, controller, dt=dt) for p in cohort]
    return compute_loss(results, dt=dt), results


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(cohort, kp_values=KP_VALUES, ki_values=KI_VALUES, kd_values=KD_VALUES, dt=DT):
    """
    Exhaustive grid search over (kp, ki, kd).

    Returns
    -------
    best_gains : dict with keys kp, ki, kd
    best_loss  : float
    all_results: list of dicts with keys kp, ki, kd, loss
    """
    grid        = list(itertools.product(kp_values, ki_values, kd_values))
    n_total     = len(grid)
    all_results = []
    best_loss   = float('inf')
    best_gains  = None

    print(f'Grid size : {n_total} candidates  ({len(kp_values)} kp × {len(ki_values)} ki × {len(kd_values)} kd)')
    print(f'Cohort    : {len(cohort)} patients')
    print(f'Simulations: {n_total * len(cohort):,}')
    print()

    t_start = time.perf_counter()

    for idx, (kp, ki, kd) in enumerate(grid):
        controller = PIDController(kp=kp, ki=ki, kd=kd, dt=dt)
        loss, _    = evaluate_closed_loop(cohort, controller, dt=dt)

        all_results.append({'kp': kp, 'ki': ki, 'kd': kd, 'loss': loss})

        if loss < best_loss:
            best_loss  = loss
            best_gains = {'kp': kp, 'ki': ki, 'kd': kd}

        # Progress every 10%
        if (idx + 1) % max(1, n_total // 10) == 0:
            elapsed  = time.perf_counter() - t_start
            per_iter = elapsed / (idx + 1)
            remaining = per_iter * (n_total - idx - 1)
            print(f'  [{idx+1:>{len(str(n_total))}}/{n_total}]  '
                  f'best={best_loss:.4f}  '
                  f'elapsed={elapsed:.1f}s  '
                  f'eta={remaining:.1f}s')

    elapsed = time.perf_counter() - t_start
    print(f'\nDone in {elapsed:.2f}s  ({elapsed/n_total*1000:.1f} ms/candidate)')
    print(f'Best gains : kp={best_gains["kp"]}  ki={best_gains["ki"]}  kd={best_gains["kd"]}')
    print(f'Best loss  : {best_loss:.4f}')

    return best_gains, best_loss, all_results, elapsed


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_grid_search(all_results, best_gains, out_path='results/grid_search.png'):
    """
    3D scatter plot of the gain space colored by loss.

    Log-scaled loss is used so that induction failures (loss ~7+) don't wash
    out the color variation in the well-behaved region.  Failed candidates are
    rendered small and semi-transparent to show where the bad region is without
    dominating the visual.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from plot_style import apply as apply_style, COLORS

    apply_style()
    C = COLORS

    kp   = [r['kp']   for r in all_results]
    ki   = [r['ki']   for r in all_results]
    kd   = [r['kd']   for r in all_results]
    loss = [r['loss'] for r in all_results]

    log_loss  = [math.log10(max(l, 1e-6)) for l in loss]
    threshold = math.log10(1.0)   # log10(1) = 0 — anything above is a failed induction

    good = [(i, l) for i, l in enumerate(log_loss) if l <= threshold]
    bad  = [(i, l) for i, l in enumerate(log_loss) if l >  threshold]

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(C['bg'])
    ax.set_facecolor(C['surface'])

    # Failed candidates — muted, small
    if bad:
        bi = [i for i, _ in bad]
        ax.scatter([kp[i] for i in bi], [ki[i] for i in bi], [kd[i] for i in bi],
                   c=C['muted'], s=20, alpha=0.3, label='Failed induction')

    # Good candidates — color-mapped by log loss
    if good:
        gi    = [i for i, _ in good]
        gvals = [l for _, l in good]
        sc = ax.scatter(
            [kp[i] for i in gi], [ki[i] for i in gi], [kd[i] for i in gi],
            c=gvals, cmap='plasma_r', s=60, alpha=0.85,
            vmin=min(gvals), vmax=max(gvals),
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label('log₁₀(loss)', color=C['muted'], fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=C['muted'])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=C['muted'])

    # Mark best point
    ax.scatter([best_gains['kp']], [best_gains['ki']], [best_gains['kd']],
               c=C['teal'], s=120, marker='*', zorder=10, label='Best')

    ax.set_xlabel('kp', color=C['muted'], fontsize=9, labelpad=8)
    ax.set_ylabel('ki', color=C['muted'], fontsize=9, labelpad=8)
    ax.set_zlabel('kd', color=C['muted'], fontsize=9, labelpad=8)
    ax.tick_params(colors=C['muted'], labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(C['border'])
    ax.yaxis.pane.set_edgecolor(C['border'])
    ax.zaxis.pane.set_edgecolor(C['border'])
    ax.grid(True, color=C['border'], linewidth=0.5)

    ax.set_title('PID gain search — loss landscape', color=C['text'], fontsize=11, pad=12)
    ax.legend(fontsize=8, facecolor=C['surface'], edgecolor=C['border'],
              labelcolor=C['text'])

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Plot saved to {out_path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid search over PID gains.')
    parser.add_argument('--cohort', type=str, default=None,
                        help='Path to saved cohort JSON. If omitted, uses n=200 seed=99 in-memory.')
    parser.add_argument('--out', type=str, default='results/grid_search.json',
                        help='Path to save grid search results.')
    args = parser.parse_args()

    if args.cohort:
        cohort = cohort_from_json(args.cohort)
        print(f'Loaded {len(cohort)}-patient cohort from {args.cohort}')
    else:
        cohort = generate_cohort(n=200, seed=99)
        print(f'Using in-memory cohort (n=200, seed=99)')

    best_gains, best_loss, all_results, elapsed = grid_search(cohort)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    output = {
        'best_gains': best_gains,
        'best_loss':  best_loss,
        'elapsed_s':  round(elapsed, 3),
        'grid':       all_results,
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to {args.out}')

    plot_grid_search(all_results, best_gains,
                     out_path=args.out.replace('.json', '.png'))
