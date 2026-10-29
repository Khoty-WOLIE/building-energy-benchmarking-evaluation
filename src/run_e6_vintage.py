"""E6 - data-vintage experiment (protocol v1.0, section 6; contrasts P6-P8; 2x2 factorial added before execution).

Same protocol on two releases of 2016: original annual release vs 2016 rows of the consolidated release,
restricted to buildings present in both. Factorial: attributes {orig, cons} x target {orig, cons}.

Usage (from article-01-seattle/):
    python src/run_e6_vintage.py data/raw/seattle_benchmarking_2015_present_teqw-tu6e.csv \
        data/raw/2016_Building_Energy_Benchmarking.csv
"""
import json
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

NUMERIC = ["log_gfa_total", "log_gfa_buildings", "parking_share", "largest_use_share", "NumberofFloors",
           "NumberofBuildings", "YearBuilt", "Latitude", "Longitude"]
CATEGORICAL = ["use_type", "Neighborhood"]
MODELS = ["M0_median_by_type", "M2_ridge", "M3_random_forest", "M4_hist_gradient_boosting"]
TARGETS = {"energy": D.TARGET, "ghg": D.TARGET_GHG}
CELLS = [("orig", "orig"), ("cons", "cons"), ("orig", "cons"), ("cons", "orig")]   # (attributes, target)


def common_features(f):
    out = pd.DataFrame({D.ID: f[D.ID].astype(str)})
    gfa = pd.to_numeric(f["PropertyGFATotal"], errors="coerce")
    out["log_gfa_total"] = np.log(gfa)
    out["log_gfa_buildings"] = np.log1p(pd.to_numeric(f["PropertyGFABuildings"], errors="coerce").clip(lower=0))
    out["parking_share"] = (pd.to_numeric(f["PropertyGFAParking"], errors="coerce").clip(lower=0) / gfa).clip(0, 1)
    out["largest_use_share"] = (pd.to_numeric(f["LargestPropertyUseTypeGFA"], errors="coerce") / gfa).clip(0, 2)
    for c in ["NumberofFloors", "NumberofBuildings", "YearBuilt", "Latitude", "Longitude"]:
        out[c] = pd.to_numeric(f[c], errors="coerce")
    out["Neighborhood"] = D.harmonise_neighborhood(f["Neighborhood"]).to_numpy()
    out["use_type_raw"] = f["LargestPropertyUseType"].fillna("UNKNOWN").str.strip().to_numpy()
    for k, col in TARGETS.items():
        out[f"y_{k}"] = pd.to_numeric(f[col], errors="coerce").to_numpy()
    return out.set_index(D.ID)


def main(cons_path, orig_path):
    run = runlog.start_run("E6", [cons_path, orig_path])
    pop, _ = D.apply_population(D.load_consolidated(cons_path))
    cons = common_features(pop[pop[D.YEAR] == 2016])
    orig_raw = pd.read_csv(orig_path, low_memory=False).rename(columns={"PropertyGFABuilding(s)": "PropertyGFABuildings"})
    orig = common_features(orig_raw)
    ids = sorted(set(cons.index) & set(orig.index))
    cons, orig = cons.loc[ids], orig.loc[ids]

    # Rare use types grouped on the union of both releases (category counts only, no target).
    counts = pd.concat([cons["use_type_raw"], orig["use_type_raw"]]).value_counts()
    keep = set(counts[counts >= 2 * D.RARE_TYPE_MIN_BUILDINGS].index)   # each building counted once per release
    for f in (cons, orig):
        f["use_type"] = f["use_type_raw"].where(f["use_type_raw"].isin(keep), "Other (grouped)")
    rel = {"orig": orig, "cons": cons}

    rows, preds, notes = [], [], {"matched_buildings": len(ids)}
    for tkey in TARGETS:
        ycol = f"y_{tkey}"
        valid = [b for b in ids if orig.at[b, ycol] > 0 and cons.at[b, ycol] > 0]
        notes[f"buildings_with_positive_{tkey}_in_both"] = len(valid)
        for attrs, target in CELLS:
            frame = rel[attrs].loc[valid, NUMERIC + CATEGORICAL].copy()
            frame["y"] = np.log(rel[target].loc[valid, ycol].to_numpy())
            frame = frame.reset_index()
            for sp in S.grouped(frame, k=5, repeats=5):
                tr, te = frame.iloc[sp["train"]], frame.iloc[sp["test"]]
                for m in MODELS:
                    p_te, p_tr, best = M.fit_predict(m, tr, te, NUMERIC, CATEGORICAL, "y", inner="grouped",
                                                     type_col="use_type")
                    smear = St.duan_factor(tr["y"], p_tr)
                    preds.append(pd.DataFrame({"target": tkey, "attrs": attrs, "y_release": target, "model": m,
                                               "repeat": sp["repeat"], "fold": sp["fold"], D.ID: te[D.ID].to_numpy(),
                                               "y": te["y"].to_numpy(), "p": p_te, "smear": smear}))
                    rows.append({"target": tkey, "attrs": attrs, "y_release": target, "model": m,
                                 "repeat": sp["repeat"], "fold": sp["fold"], "best_params": json.dumps(best),
                                 **St.point_metrics(te["y"], p_te, smear)})
            print(f"done {tkey} attrs={attrs} target={target}", flush=True)

    P = pd.concat(preds, ignore_index=True)
    P.to_csv(run / "predictions_e6.csv.gz", index=False)
    pd.DataFrame(rows).to_csv(run / "metrics_by_fold.csv", index=False)
    (run / "notes.json").write_text(json.dumps(notes, indent=2))
    analyze(run)


def analyze(run):
    """Summaries and contrasts from the saved predictions (re-runnable without refitting models)."""
    run = Path(run)
    P = pd.read_csv(run / "predictions_e6.csv.gz", dtype={D.ID: str})
    notes = json.loads((run / "notes.json").read_text())
    # Per-building absolute error averaged over the 5 repeats, then cluster (= building) bootstrap.
    P["abs_err"] = (P["p"] - P["y"]).abs()
    per_b = P.groupby(["target", "attrs", "y_release", "model", D.ID])["abs_err"].mean().rename("err").reset_index()
    summary = []
    for key, g in per_b.groupby(["target", "attrs", "y_release", "model"]):
        est, lo, hi, _ = St.cluster_bootstrap_mean(g["err"], g[D.ID])
        summary.append(dict(zip(["target", "attrs", "y_release", "model"], key),
                            mae_log=est, ci_low=lo, ci_high=hi, n_buildings=len(g)))
    pd.DataFrame(summary).to_csv(run / "summary_mae.csv", index=False)

    def err(t, a, y, m):
        g = per_b[(per_b["target"] == t) & (per_b["attrs"] == a) & (per_b["y_release"] == y) & (per_b["model"] == m)]
        return g.set_index(D.ID)["err"]

    contrasts = {}
    spec = {"P6": ("ghg", ("orig", "orig"), ("cons", "cons")),
            "P7": ("energy", ("orig", "orig"), ("cons", "cons")),
            "P8": ("energy", ("orig", "cons"), ("cons", "cons")),
            "X_target_effect_energy": ("energy", ("cons", "orig"), ("cons", "cons")),
            "X_target_effect_ghg": ("ghg", ("cons", "orig"), ("cons", "cons")),
            "X_attr_effect_ghg": ("ghg", ("orig", "cons"), ("cons", "cons"))}
    for name, (t, a, b) in spec.items():
        for m in MODELS:
            ea, eb = err(t, *a, m), err(t, *b, m)
            common = ea.index.intersection(eb.index)
            r = St.paired_contrast(ea[common], eb[common], common.to_numpy())
            contrasts[(name, m)] = {k: v for k, v in r.items() if k != "boot"} | {"n_buildings": len(common)}
    C = pd.DataFrame([{"contrast": k[0], "model": k[1], **v} for k, v in contrasts.items()])
    C["primary"] = C["contrast"].isin(["P6", "P7", "P8"]) & C["model"].eq("M4_hist_gradient_boosting")
    C.to_csv(run / "contrasts.csv", index=False)

    rank = {}
    S_ = pd.DataFrame(summary)
    for t in TARGETS:
        a = S_[(S_["target"] == t) & (S_["attrs"] == "orig") & (S_["y_release"] == "orig")].set_index("model")["mae_log"]
        b = S_[(S_["target"] == t) & (S_["attrs"] == "cons") & (S_["y_release"] == "cons")].set_index("model")["mae_log"]
        rank[t] = {"spearman_rank_corr": float(a.rank().corr(b.rank(), method="spearman")),
                   "order_orig": list(a.sort_values().index), "order_cons": list(b.sort_values().index)}
    notes["ranking_stability"] = rank
    (run / "notes.json").write_text(json.dumps(notes, indent=2))
    print(run)
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(C.drop(columns=[]).round(4).to_string(index=False))
    print(json.dumps(notes, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--analyze":
        analyze(sys.argv[2])
    else:
        main(*sys.argv[1:3])
