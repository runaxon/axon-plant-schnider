"""
Parallel grid search using the closed-form analytical C extension.

Ad, Bd, and Ce coefficients are precomputed once per patient in the pool
initializer. Workers receive only (kp, ki, kd) tuples per task.

Usage
-----
    python grid_search_analytical.py
    python grid_search_analytical.py --workers 14
"""

import argparse
import itertools
import multiprocessing as mp
import time

from physio.cohort import cohort_from_json
from physio.schnider import params_from_patient
from eval_cohort import DURATION, DT, BIS_TARGET, T_INDUCTION, T_MAINTENANCE, ISE_REF, LAMBDA_NORM

N          = 8
KP_CENTER  = 4000
KI_CENTER  = 400
KD_CENTER  = 1600
KP_VALUES  = [2 * KP_CENTER / N * x for x in range(1, N + 1)]
KI_VALUES  = [2 * KI_CENTER / N * x for x in range(1, N + 1)]
KD_VALUES  = [2 * KD_CENTER / N * x for x in range(1, N + 1)]

_cohort   = None
_matrices = None

def _init_worker(cohort, matrices):
    global _cohort, _matrices
    _cohort   = cohort
    _matrices = matrices


def _evaluate(args):
    from cython_ext.schnider_analytical_cy import simulate_closed_loop_analytical
    kp, ki, kd = args
    ises = [
        simulate_closed_loop_analytical(
            p, kp, ki, kd,
            m['Ad'], m['Bd'], m['alpha'], m['h_coeffs'], m['p_coeffs'], m['Vinv'],
            duration=DURATION, dt=DT, bis_target=BIS_TARGET,
            t_induction=T_INDUCTION, t_maintenance=T_MAINTENANCE,
        )
        for p, m in zip(_cohort, _matrices)
    ]
    loss = LAMBDA_NORM * sum(ises) / (len(_cohort) * ISE_REF)
    return loss, kp, ki, kd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort',  default='cohorts/n200_seed99.json')
    parser.add_argument('--workers', type=int, default=min(14, mp.cpu_count()))
    args = parser.parse_args()

    from cython_ext.schnider_analytical_cy import precompute_matrices

    cohort   = cohort_from_json(args.cohort)
    matrices = [precompute_matrices(params_from_patient(p), DT) for p in cohort]

    candidates = list(itertools.product(KP_VALUES, KI_VALUES, KD_VALUES))
    n = len(candidates)

    print(f'Grid: {n} candidates  |  Workers: {args.workers}')

    t0 = time.perf_counter()

    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(cohort, matrices),
    ) as pool:
        results = pool.map(_evaluate, candidates, chunksize=max(1, n // (args.workers * 4)))

    elapsed = time.perf_counter() - t0

    best_loss, best_kp, best_ki, best_kd = min(results, key=lambda r: r[0])

    print(f'Done in {elapsed:.3f}s  ({elapsed/n*1000:.2f} ms/candidate)')
    print(f'Best gains: kp={best_kp:.0f}  ki={best_ki:.0f}  kd={best_kd:.0f}')
    print(f'Best loss:  {best_loss:.2e}')
