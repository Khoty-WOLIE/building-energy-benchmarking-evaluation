"""Exploratory analyses X1 (fingerprint test) and X2 (T1 last-year baseline). Decided AFTER the primary
results (protocol section 12); reported as exploratory, outside the Holm family.

Usage (from article-01-seattle/):
    python src/run_exploratory.py X1
    python src/run_exploratory.py X2
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a01 import data as D  # noqa: E402
from a01 import models as M  # noqa: E402
from a01 import runlog  # noqa: E402
from a01 import splits as S  # noqa: E402
from a01 import stats as St  # noqa: E402
from run_panel import CONS, base_frame, with_t1  # noqa: E402

SEED = 20261029


def jitter(d, rng):
    """Independent per-row noise on continuous building attributes: breaks exact building identification."""
    j = d.copy()
    n = len(j)
    j["log_gfa_total"] = j["log_gfa_total"] + np.log1p(rng.normal(0, 0.02, n))
    j["log_gfa_buildings"] = j["log_gfa_buildings"] + np.log1p(rng.normal(0, 0.02, n))
    j["Latitude"] = j["Latitude"] + rng.normal(0, 0.002, n)
    j["Longitude"] = j["Longitude"] + rng.normal(0, 0.002, n)
    j["YearBuilt"] = j["YearBuilt"] + rng.randint(-2, 3, n)
    j["parking_share"] = (j["parking_share"] * (1 + rng.normal(0, 0.02, n))).clip(0, 1)
    j["largest_use_share"] = (j["largest_use_share"] * (1 + rng.normal(0, 0.02, n))).clip(0, 2)
    return j


def x1(run):
    d = with_t1(base_frame(), None)
    # How identifying are the T0 attributes? Share of buildings whose T0 vector is unique in the panel.
    key = d[D.T0_NUMERIC + D.T0_CATEGORICAL].round(6).astype(str).agg("|".join, axis=1)
    per_b = key.groupby(d[D.ID]).first()
    uniq = float((per_b.map(per_b.value_counts()) == 1).mean())
    dj = jitter(d, np.random.RandomState(SEED))
    rows, preds = [], []
    for variant, frame in (("exact", d), ("jittered", dj)):
        for gen, inner in ((S.row_random(frame, repeats=2), "row_random"), (S.grouped(frame, repeats=2), "grouped")):
            for sp in gen:
                tr, te = frame.iloc[sp["train"]], frame.iloc[sp["test"]]
                for m in ("M2_ridge", "M3_random_forest", "M4_hist_gradient_boosting"):
                    p, _, _ = M.fit_predict(m, tr, te, D.T0_NUMERIC, D.T0_CATEGORICAL, "y_log", inner=inner)
                    preds.append(pd.DataFrame({"variant": variant, "split": sp["split"], "repeat": sp["repeat"],
                                               "model": m, D.ID: te[D.ID].to_numpy(), D.YEAR: te[D.YEAR].to_numpy(),
                                               "abs_err": np.abs(p - te["y_log"].to_numpy())}))
            print("X1", variant, sp["split"], "done", flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(run / "predictions_X1.csv.gz", index=False)
    per_row = P.groupby(["variant", "split", "model", D.ID, D.YEAR])["abs_err"].mean().rename("err").reset_index()
    for (v, s, m), g in per_row.groupby(["variant", "split", "model"]):
        est, lo, hi, _ = St.cluster_bootstrap_mean(g["err"], g[D.ID])
        rows.append({"variant": v, "split": s, "model": m, "mae_log": est, "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(run / "summary_X1.csv", index=False)
    (run / "notes.txt").write_text(f"share_of_buildings_with_unique_T0_vector={uniq:.4f}\n")
    print(f"unique T0 vector share: {uniq:.4f}")
    print(out.round(4).to_string(index=False))


def t1_last(base, cutoff):
    """T1 with the most recent observed year strictly before the row's year, never later than `cutoff`.

    Training rows (year <= cutoff) use their previous observed year; test rows (year > cutoff) use the last year
    <= cutoff. Same row order as with_t1() (sorted by building, year).
    """
    f = base.sort_values([D.ID, D.YEAR]).reset_index(drop=True)
    cols = [D.YEAR, D.TARGET] + D.MIX
    prev = f.groupby(D.ID)[cols].shift(1)
    last = f[f[D.YEAR] <= cutoff].groupby(D.ID)[cols].last()
    future = f[D.YEAR] > cutoff
    for c in cols:
        prev.loc[future, c] = f.loc[future, D.ID].map(last[c]).to_numpy()
    tot = prev[D.MIX].fillna(0).sum(axis=1).replace(0, np.nan)
    f["ref_year"] = prev[D.YEAR]
    f["ref_share_electricity"] = prev["Electricity(kBtu)"].fillna(0) / tot
    f["ref_share_gas"] = prev["NaturalGas(kBtu)"].fillna(0) / tot
    f["ref_share_steam"] = prev["SteamUse(kBtu)"].fillna(0) / tot
    f["ref_log_energy"] = np.log(prev[D.TARGET])
    f["ref_gap_years"] = f[D.YEAR] - f["ref_year"]
    f["has_ref"] = f["ref_year"].notna() & f["ref_share_electricity"].notna()
    f["persist_log"] = f["ref_log_energy"]           # persistence = the same last-year baseline
    return f


def x2(run):
    base = base_frame()
    rows, preds = [], []
    frames = {}
    for sp in S.grouped_temporal(with_t1(base, None)):
        c = sp["cutoff"]
        if c not in frames:
            frames[c] = t1_last(base, c)
        f = frames[c]
        tr, te = f.iloc[sp["train"]], f.iloc[sp["test"]]
        tr, te = tr[tr["has_ref"]], te[te["has_ref"]]
        for m in ("M1_persistence", "M2_ridge", "M4_hist_gradient_boosting"):
            p, _, _ = M.fit_predict(m, tr, te, D.T0_NUMERIC + D.T1_EXTRA, D.T0_CATEGORICAL, "y_log", inner="grouped")
            preds.append(pd.DataFrame({"horizon": sp["horizon"], "cutoff": c, "model": m, D.ID: te[D.ID].to_numpy(),
                                       D.YEAR: te[D.YEAR].to_numpy(), "abs_err": np.abs(p - te["y_log"].to_numpy())}))
        print("X2", sp["cutoff"], sp["fold"], sp["horizon"], "done", flush=True)
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(run / "predictions_X2.csv.gz", index=False)
    for (m, h), g in P.groupby(["model", "horizon"]):
        est, lo, hi, _ = St.cluster_bootstrap_mean(g["abs_err"], g[D.ID])
        rows.append({"model": m, "horizon": h, "mae_log": est, "ci_low": lo, "ci_high": hi, "n_rows": len(g)})
    for m in ("M1_persistence", "M2_ridge", "M4_hist_gradient_boosting"):
        g = P[P.model == m]
        est, lo, hi, _ = St.cluster_bootstrap_mean(g["abs_err"], g[D.ID])
        rows.append({"model": m, "horizon": "all", "mae_log": est, "ci_low": lo, "ci_high": hi, "n_rows": len(g)})
    out = pd.DataFrame(rows)
    out.to_csv(run / "summary_X2.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1]
    run = runlog.start_run(which, [CONS])
    {"X1": x1, "X2": x2}[which](run)
    print(run)
