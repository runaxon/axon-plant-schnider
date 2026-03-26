"""
Fast grid search using the full C extension.

Identical grid and cohort as grid_search.py — drop-in comparison.
Runs the same 512 candidates x 200 patients, prints wall-clock time.

Usage
-----
    python grid_search_fast.py
    python grid_search_fast.py --cohort cohorts/n200_seed99.json
"""

import argparse
import itertools
import time

import numpy as np

from physio.cohort import cohort_from_json
from cython_ext.schnider_full_cy import simulate_closed_loop_cy
from eval_cohort import DURATION, DT, BIS_TARGET, T_INDUCTION, T_MAINTENANCE, ISE_REF, LAMBDA_NORM

# Same grid as grid_search.py
N          = 8
KP_CENTER  = 4000
KI_CENTER  = 400
KD_CENTER  = 1600
KP_VALUES  = [2 * KP_CENTER / N * x for x in range(1, N + 1)]
KI_VALUES  = [2 * KI_CENTER / N * x for x in range(1, N + 1)]
KD_VALUES  = [2 * KD_CENTER / N * x for x in range(1, N + 1)]


def grid_search_fast(cohort):
    candidates = list(itertools.product(KP_VALUES, KI_VALUES, KD_VALUES))
    n = len(candidates)
    print(f'Grid: {n} candidates  |  Cohort: {len(cohort)} patients')
    print()

    best_loss   = float('inf')
    best_gains  = None
    t0          = time.perf_counter()

    for i, (kp, ki, kd) in enumerate(candidates):
        ises = [
            simulate_closed_loop_cy(
                p, kp=kp, ki=ki, kd=kd,
                duration=DURATION, dt=DT,
                bis_target=BIS_TARGET,
                t_induction=T_INDUCTION,
                t_maintenance=T_MAINTENANCE,
            )
            for p in cohort
        ]
        loss = LAMBDA_NORM * sum(ises) / (len(cohort) * ISE_REF)

        if loss < best_loss:
            best_loss  = loss
            best_gains = (kp, ki, kd)

        if (i + 1) % 64 == 0:
            elapsed = time.perf_counter() - t0
            rate    = (i + 1) / elapsed
            eta     = (n - i - 1) / rate
            print(f'  {i+1:4d}/{n}  elapsed {elapsed:.1f}s  ETA {eta:.1f}s  '
                  f'best loss {best_loss:.2e}')

    elapsed = time.perf_counter() - t0
    print()
    print(f'Done in {elapsed:.2f}s  ({elapsed/n*1000:.1f} ms/candidate)')
    print(f'Best gains: kp={best_gains[0]:.0f}  ki={best_gains[1]:.0f}  kd={best_gains[2]:.0f}')
    print(f'Best loss:  {best_loss:.2e}')
    return best_gains, best_loss, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', default='cohorts/n200_seed99.json')
    args = parser.parse_args()

    cohort = cohort_from_json(args.cohort)
    grid_search_fast(cohort)
