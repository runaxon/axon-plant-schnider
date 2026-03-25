"""
physio.cohort
=============

Virtual patient cohort generation via Latin Hypercube Sampling (LHS).

LHS guarantees uniform coverage across every demographic dimension with far
fewer samples than pure random sampling.  For N patients and d continuous
variables, the algorithm:

    1. Divides each dimension into N equal-probability intervals.
    2. Samples exactly once per interval (with uniform jitter within it).
    3. Permutes the samples independently across dimensions.

Each column is then scaled to its demographic range [a, b].  Sex is handled
separately: floor(N/2) males and ceil(N/2) females, then shuffled.

Reference: McKay, Beckman, Conover (1979). Technometrics 21(2):239-245.
"""

from __future__ import annotations

import json
import random

from physio.core import Patient

# Demographic ranges used for sampling.
# Chosen to reflect the adult surgical population studied in Schnider (1998).
DEMOGRAPHIC_RANGES: dict[str, tuple[float, float]] = {
    'age': (18.0, 80.0),  # years
    'weight': (50.0, 120.0),  # kg
    'height': (150.0, 195.0)  # cm
}

def lhs(
    n: int,
    ranges: dict[str, tuple[float, float]],
    seed: int | None = None,
) -> dict[str, list[float]]:
    """
    Generate an LHS matrix for the given continuous variables.

    Parameters
    ----------
    n: Number of samples (patients).
    ranges: Dict mapping variable name to (min, max).
    seed: Random seed for reproducibility.

    Returns
    -------
    Dict mapping each variable name to a list of n sampled values.
    """
    rng = random.Random(seed)

    samples: dict[str, list[float]] = {}
    for name, (lo, hi) in ranges.items():
        # Build permuted interval indices: one sample per stratum
        perm = list(range(n))
        rng.shuffle(perm)

        # Sample with uniform jitter inside each stratum, scale to [lo, hi]
        col = [
            lo + (perm[i] + rng.random()) / n * (hi - lo)
            for i in range(n)
        ]
        samples[name] = col

    return samples

def generate_cohort(
    n: int,
    ranges: dict[str, tuple[float, float]] | None = None,
    seed: int | None = None,
) -> list[Patient]:
    """
    Generate a virtual cohort of N patients via Latin Hypercube Sampling.

    Continuous demographics (age, weight, height) are sampled with LHS.
    Sex is assigned as floor(N/2) males + ceil(N/2) females, then shuffled.

    Parameters
    ----------
    n: Cohort size.
    ranges:
        Override the default demographic ranges. Keys must be a subset of
        { 'age', 'weight', 'height' }.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    List of N Patient instances.
    """
    if ranges is None:
        ranges = DEMOGRAPHIC_RANGES

    samples = lhs(n, ranges, seed=seed)

    # Sex assignment: balanced split, shuffled
    rng = random.Random(seed)
    n_male = n // 2
    n_female = n - n_male
    sexes = ['male'] * n_male + ['female'] * n_female
    rng.shuffle(sexes)

    cohort = [
        Patient(
            age = round(samples['age'][i], 1),
            weight = round(samples['weight'][i], 1),
            height = round(samples['height'][i], 1),
            sex = sexes[i]
        )
        for i in range(n)
    ]

    return cohort


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def cohort_to_json(cohort: list[Patient], path: str) -> None:
    """
    Serialize a cohort to a JSON file.

    Each patient is stored as a plain object with keys:
    age, weight, height, sex.

    Parameters
    ----------
    cohort : list of Patient instances to persist.
    path   : destination file path (e.g. 'cohorts/n100_seed42.json').
    """
    records = [
        {'age': p.age, 'weight': p.weight, 'height': p.height, 'sex': p.sex}
        for p in cohort
    ]
    with open(path, 'w') as f:
        json.dump(records, f, indent=2)


def cohort_from_json(path: str) -> list[Patient]:
    """
    Load a cohort previously saved with cohort_to_json().

    Parameters
    ----------
    path : path to the JSON file.

    Returns
    -------
    List of Patient instances in the original order.
    """
    with open(path) as f:
        records = json.load(f)
    return [
        Patient(age=r['age'], weight=r['weight'], height=r['height'], sex=r['sex'])
        for r in records
    ]
