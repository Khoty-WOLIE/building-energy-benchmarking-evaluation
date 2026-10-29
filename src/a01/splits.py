"""Evaluation splits (protocol v1.0, section 4). Each generator yields dicts with train/test row positions.

S1 row_random      : K-fold over building-years (optimistic reference; same building on both sides).
S2 grouped         : K-fold over buildings (no building in both train and test).
S3 forward_temporal: train on years <= cutoff, test on year cutoff + h.
S4 grouped_temporal: S3 restricted so that test buildings never appear in training.
"""
import numpy as np

from .data import ID, YEAR


def _building_folds(buildings, k, rng):
    b = np.array(sorted(buildings))
    rng.shuffle(b)
    return {bid: i % k for i, bid in enumerate(b)}


def row_random(df, k=5, repeats=5, seed=20261029):
    for r in range(repeats):
        rng = np.random.RandomState(seed + r)
        fold = rng.permutation(len(df)) % k
        for f in range(k):
            yield {"split": "S1_row_random", "repeat": r, "fold": f,
                   "train": np.where(fold != f)[0], "test": np.where(fold == f)[0]}


def grouped(df, k=5, repeats=5, seed=20261029):
    ids = df[ID].to_numpy()
    for r in range(repeats):
        fmap = _building_folds(set(ids), k, np.random.RandomState(seed + r))
        fold = np.array([fmap[b] for b in ids])
        for f in range(k):
            yield {"split": "S2_grouped", "repeat": r, "fold": f,
                   "train": np.where(fold != f)[0], "test": np.where(fold == f)[0]}


def forward_temporal(df, cutoffs=(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024), max_horizon=3):
    years = df[YEAR].to_numpy()
    for t in cutoffs:
        train = np.where(years <= t)[0]
        for h in range(1, max_horizon + 1):
            test = np.where(years == t + h)[0]
            if len(test):
                yield {"split": "S3_forward_temporal", "cutoff": t, "horizon": h, "test_year": t + h,
                       "train": train, "test": test}


def grouped_temporal(df, cutoffs=(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024), max_horizon=3, k=5,
                     seed=20261029):
    years, ids = df[YEAR].to_numpy(), df[ID].to_numpy()
    fmap = _building_folds(set(ids), k, np.random.RandomState(seed))
    fold = np.array([fmap[b] for b in ids])
    for t in cutoffs:
        for f in range(k):
            train = np.where((years <= t) & (fold != f))[0]
            for h in range(1, max_horizon + 1):
                test = np.where((years == t + h) & (fold == f))[0]
                if len(test):
                    yield {"split": "S4_grouped_temporal", "cutoff": t, "horizon": h, "test_year": t + h,
                           "fold": f, "train": train, "test": test}
