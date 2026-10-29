import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from a01 import models as M  # noqa: E402
from a01 import stats as St  # noqa: E402


def test_point_metrics_perfect_prediction():
    y = np.log(np.array([10.0, 100.0, 1000.0]))
    m = St.point_metrics(y, y)
    assert m["mae_log"] == 0 and m["r2_log"] == 1 and m["mdape"] == 0


def test_duan_factor_is_one_without_residuals():
    y = np.array([1.0, 2.0])
    assert St.duan_factor(y, y) == 1.0


def test_cluster_bootstrap_treats_clusters_as_units():
    values = np.array([1.0] * 100 + [0.0])        # one big cluster of ones, one singleton zero
    clusters = ["a"] * 100 + ["b"]
    est, lo, hi, _ = St.cluster_bootstrap_mean(values, clusters, B=500)
    assert np.isclose(est, 100 / 101)
    assert lo == 0.0 or lo < 0.5                  # resampling clusters can drop cluster "a" entirely


def test_paired_contrast_detects_shift():
    rng = np.random.RandomState(1)
    a = rng.rand(500) + 0.1
    b = a - 0.1
    r = St.paired_contrast(a, b, np.arange(500), B=500)
    assert np.isclose(r["estimate"], 0.1) and r["ci_low"] > 0


def test_holm_monotone_and_bounded():
    adj = St.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06


def test_median_by_group_handles_unseen_level():
    X = pd.DataFrame({"property_type": ["A", "A", "B"]})
    m = M.MedianByGroup().fit(X, [1.0, 3.0, 10.0])
    assert list(m.predict(pd.DataFrame({"property_type": ["A", "Z"]}))) == [2.0, 3.0]
