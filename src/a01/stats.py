"""Metrics and cluster-bootstrap inference (protocol v1.0, sections 7-8)."""
import numpy as np
import pandas as pd

B_DEFAULT = 2000


def point_metrics(y, p, smear=1.0):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    ey, ep = np.exp(y), np.exp(p)
    return {"mae_log": float(np.abs(e).mean()), "rmse_log": float(np.sqrt((e ** 2).mean())),
            "r2_log": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
            "mdape": float(np.median(np.abs(ep - ey) / ey)),
            "mae_kbtu_duan": float(np.abs(ep * smear - ey).mean()), "n": int(len(y))}


def duan_factor(y_train, p_train):
    return float(np.mean(np.exp(np.asarray(y_train, float) - np.asarray(p_train, float))))


def _cluster_index(clusters):
    codes, uniq = pd.factorize(pd.Series(clusters))
    members = [np.where(codes == k)[0] for k in range(len(uniq))]
    return members


def cluster_bootstrap_mean(values, clusters, B=B_DEFAULT, seed=20261029):
    """Mean of `values` with a percentile CI from resampling whole clusters (buildings)."""
    v = np.asarray(values, float)
    members = _cluster_index(clusters)
    sums = np.array([v[m].sum() for m in members])
    sizes = np.array([len(m) for m in members])
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(members), size=(B, len(members)))
    boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), boot


def paired_contrast(err_a, err_b, clusters, B=B_DEFAULT, seed=20261029):
    """Mean(err_a - err_b) on identical rows; CI and two-sided bootstrap p-value from cluster resampling."""
    d = np.asarray(err_a, float) - np.asarray(err_b, float)
    if len(d) == 0:
        raise ValueError("paired_contrast called on an empty set of rows")
    est, lo, hi, boot = cluster_bootstrap_mean(d, clusters, B, seed)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "p_boot": float(min(1.0, max(p, 1.0 / B))),
            "boot": boot}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (dict in, dict out)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def holm_ci(boot, alpha_adjusted):
    return float(np.percentile(boot, 100 * alpha_adjusted / 2)), float(np.percentile(boot, 100 * (1 - alpha_adjusted / 2)))
