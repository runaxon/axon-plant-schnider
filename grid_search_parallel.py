"""
Parallel grid search using multiprocessing + the full C extension.

Each worker process holds the cohort in memory (loaded once via pool
initializer). The main process sends only (kp, ki, kd) tuples — no
patient data crosses the IPC boundary on every task.

Usage
-----
    python grid_search_parallel.py
    python grid_search_parallel.py --workers 8
    python grid_search_parallel.py --cohort cohorts/n200_seed99.json --workers 14
"""

import argparse
import itertools
import multiprocessing as mp
import time

from physio.cohort import cohort_from_json
from eval_cohort import DURATION, DT, BIS_TARGET, T_INDUCTION, T_MAINTENANCE, ISE_REF, LAMBDA_NORM

# Same grid as grid_search.py and grid_search_fast.py
N          = 8
KP_CENTER  = 4000
KI_CENTER  = 400
KD_CENTER  = 1600
KP_VALUES  = [2 * KP_CENTER / N * x for x in range(1, N + 1)]
KI_VALUES  = [2 * KI_CENTER / N * x for x in range(1, N + 1)]
KD_VALUES  = [2 * KD_CENTER / N * x for x in range(1, N + 1)]


# ---------------------------------------------------------------------------
# Worker state — loaded once per process, not re-pickled per task
# ---------------------------------------------------------------------------

_cohort = None

def _init_worker(cohort_path):
    global _cohort
    _cohort = cohort_from_json(cohort_path)


def _evaluate(args):
    """Evaluate one (kp, ki, kd) candidate. Returns (loss, kp, ki, kd)."""
    from cython_ext.schnider_full_cy import simulate_closed_loop_cy
    kp, ki, kd = args
    ises = [
        simulate_closed_loop_cy(
            p, kp=kp, ki=ki, kd=kd,
            duration=DURATION, dt=DT,
            bis_target=BIS_TARGET,
            t_induction=T_INDUCTION,
            t_maintenance=T_MAINTENANCE,
        )
        for p in _cohort
    ]
    loss = LAMBDA_NORM * sum(ises) / (len(_cohort) * ISE_REF)
    return loss, kp, ki, kd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def grid_search_parallel(cohort_path, n_workers):
    candidates = list(itertools.product(KP_VALUES, KI_VALUES, KD_VALUES))
    n = len(candidates)

    print(f'Grid: {n} candidates  |  Workers: {n_workers}')
    print()

    t0 = time.perf_counter()

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(cohort_path,),
    ) as pool:
        results = pool.map(_evaluate, candidates, chunksize=max(1, n // (n_workers * 4)))

    elapsed = time.perf_counter() - t0

    best_loss, best_kp, best_ki, best_kd = min(results, key=lambda r: r[0])

    print(f'Done in {elapsed:.2f}s  ({elapsed/n*1000:.1f} ms/candidate)')
    print(f'Best gains: kp={best_kp:.0f}  ki={best_ki:.0f}  kd={best_kd:.0f}')
    print(f'Best loss:  {best_loss:.2e}')
    return (best_kp, best_ki, best_kd), best_loss, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort',  default='cohorts/n200_seed99.json')
    parser.add_argument('--workers', type=int, default=min(8, mp.cpu_count()))
    args = parser.parse_args()

    grid_search_parallel(args.cohort, args.workers)
