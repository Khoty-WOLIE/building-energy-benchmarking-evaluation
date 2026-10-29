"""E1/E2/E3/E5 on the 2015-2025 panel (protocol v1.0, sections 4-6). Writes one run directory per experiment.

Usage (from article-01-seattle/):
    python src/run_panel.py E1 [--limit N]     # S1 vs S2, tiers T0/T1
    python src/run_panel.py E2 [--limit N]     # S3 forward temporal, T0/T1 (+ persistence)
    python src/run_panel.py E3 [--limit N]     # S4 grouped x temporal, T0/T1 (+ persistence)
    python src/run_panel.py E5 [--limit N]     # ENERGY STAR leakage tiers on S2 and S4
--limit runs only the first N split instances per split type (timing probe).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a01 import data as D  # noqa: E402
from a01 import models as M  # noqa: E402
from a01 import runlog  # noqa: E402
from a01 import splits as S  # noqa: E402
from a01 import stats as St  # noqa: E402

CONS = "data/raw/seattle_benchmarking_2015_present_teqw-tu6e.csv"
TUNED = ["M2_ridge", "M3_random_forest", "M4_hist_gradient_boosting"]
T0_NUM, CAT = D.T0_NUMERIC, D.T0_CATEGORICAL
TIERS = {
    "T0": (T0_NUM, "all"),
    "T1": (T0_NUM + D.T1_EXTRA, "has_ref"),
    "T0+ES": (T0_NUM + ["ENERGYSTARScore", "es_missing"], "all"),
    "T1+ES": (T0_NUM + D.T1_EXTRA + ["ENERGYSTARScore", "es_missing"], "has_ref"),
    "T0+EUI": (T0_NUM + ["SiteEUI(kBtu/sf)", "SourceEUI(kBtu/sf)"], "all"),
}
PLAN = {
    "E1": {"splits": ["S1", "S2"], "tiers": ["T0", "T1"], "models": ["M0_median_by_type"] + TUNED},
    "E2": {"splits": ["S3"], "tiers": ["T0", "T1"], "models": ["M0_median_by_type", "M1_persistence"] + TUNED},
    "E3": {"splits": ["S4"], "tiers": ["T0", "T1"], "models": ["M0_median_by_type", "M1_persistence"] + TUNED},
    "E5": {"splits": ["S2", "S4"], "tiers": ["T0", "T0+ES", "T1", "T1+ES", "T0+EUI"],
           "models": ["M0_median_by_type", "M2_ridge", "M4_hist_gradient_boosting"]},
}


def base_frame():
    pop, _ = D.apply_population(D.load_consolidated(CONS))
    d = D.add_t0_features(pop, D.property_type_map(pop)).reset_index(drop=True)
    d["es_missing"] = d["ENERGYSTARScore"].isna().astype(int)
    d["es_present"] = ~d["ENERGYSTARScore"].isna()
    return d


def with_t1(d, cutoff):
    """T1 features + persistence input, both restricted to information available at `cutoff` (None = no cap)."""
    f = D.add_t1_features(d, max_ref_year=cutoff)
    f = f.sort_values([D.ID, D.YEAR]).reset_index(drop=True)
    if cutoff is None:
        f["persist_log"] = f.groupby(D.ID)["y_log"].shift(1)
    else:
        last = f[f[D.YEAR] <= cutoff].groupby(D.ID)["y_log"].last()
        f["persist_log"] = np.where(f[D.YEAR] > cutoff, f[D.ID].map(last), f.groupby(D.ID)["y_log"].shift(1))
    return f


def split_iter(code, d, limit):
    gen = {"S1": lambda: S.row_random(d), "S2": lambda: S.grouped(d),
           "S3": lambda: S.forward_temporal(d), "S4": lambda: S.grouped_temporal(d)}[code]()
    for i, sp in enumerate(gen):
        if limit is not None and i >= limit:
            return
        yield sp


def main(exp, limit):
    plan = PLAN[exp]
    run = runlog.start_run(exp if limit is None else f"{exp}_probe", [CONS])
    base = base_frame()
    frames = {None: with_t1(base, None)}
    preds, rows, timing = [], [], []
    for code in plan["splits"]:
        temporal = code in ("S3", "S4")
        inner = "row_random" if code == "S1" else "grouped"
        ref = frames[None]
        for sp in split_iter(code, ref, limit):
            cutoff = sp.get("cutoff")
            if temporal and cutoff not in frames:
                frames[cutoff] = with_t1(base, cutoff)
            d = frames[cutoff if temporal else None]
            tr_all, te = d.iloc[sp["train"]], d.iloc[sp["test"]]
            meta = {k: sp.get(k) for k in ("split", "repeat", "fold", "cutoff", "horizon", "test_year")}
            for tier in plan["tiers"]:
                num, rows_rule = TIERS[tier]
                tr = tr_all[tr_all["has_ref"]] if rows_rule == "has_ref" else tr_all
                te_t = te[te["has_ref"]] if rows_rule == "has_ref" else te
                if len(te_t) == 0 or len(tr) == 0:
                    continue
                for m in plan["models"]:
                    if m == "M1_persistence" and rows_rule != "has_ref":
                        continue
                    t0 = time.time()
                    p_te, p_tr, best = M.fit_predict(m, tr, te_t, num, CAT, "y_log", inner=inner)
                    smear = St.duan_factor(tr["y_log"], p_tr) if m != "M1_persistence" else 1.0
                    timing.append({**meta, "tier": tier, "model": m, "seconds": time.time() - t0, "n_train": len(tr)})
                    preds.append(pd.DataFrame({**{k: v for k, v in meta.items()}, "tier": tier, "model": m,
                                               D.ID: te_t[D.ID].to_numpy(), D.YEAR: te_t[D.YEAR].to_numpy(),
                                               "y": te_t["y_log"].to_numpy(), "p": p_te, "smear": smear,
                                               "has_ref": te_t["has_ref"].to_numpy(),
                                               "es_present": te_t["es_present"].to_numpy(),
                                               "property_type": te_t["property_type"].to_numpy(),
                                               "log_gfa_total": te_t["log_gfa_total"].to_numpy(),
                                               "YearBuilt": te_t["YearBuilt"].to_numpy(),
                                               "Neighborhood": te_t["Neighborhood"].to_numpy()}))
                    rows.append({**meta, "tier": tier, "model": m, "best_params": json.dumps(best),
                                 **St.point_metrics(te_t["y_log"], p_te, smear)})
            print(f"{exp} {code} {meta} done ({time.strftime('%H:%M:%S')})", flush=True)
    pd.concat(preds, ignore_index=True).to_csv(run / f"predictions_{exp}.csv.gz", index=False)
    pd.DataFrame(rows).to_csv(run / "metrics_by_instance.csv", index=False)
    T = pd.DataFrame(timing)
    T.to_csv(run / "timing.csv", index=False)
    print(run)
    print(T.groupby(["model", "tier"])["seconds"].agg(["count", "mean", "sum"]).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=list(PLAN))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.exp, a.limit)
