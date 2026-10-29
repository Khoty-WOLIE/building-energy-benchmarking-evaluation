"""Phase 04 analysis: primary contrasts P1-P8 with Holm correction, result tables and E4 heterogeneity.

Reads the latest run directory of each experiment under results/experiments/ and writes results/phase04/.

Usage (from article-01-seattle/):
    python src/analyze_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a01 import stats as St  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "phase04"
THRESHOLD = 0.02
ID = "OSEBuildingID"
M4, M2 = "M4_hist_gradient_boosting", "M2_ridge"


def latest(exp):
    runs = sorted(p for p in (EXP / exp).glob("*") if p.is_dir())
    if not runs:
        raise SystemExit(f"no run for {exp}")
    return runs[-1]


def load_preds(exp):
    d = latest(exp)
    p = pd.read_csv(next(d.glob("predictions_*.csv.gz")), low_memory=False)
    p[ID] = p[ID].astype(str)
    p["abs_err"] = (p["p"] - p["y"]).abs()
    return p, d


def per_row(p, keys):
    """Average absolute error per test row over CV repeats (rows are identified by building + year)."""
    return p.groupby(keys + [ID, "DataYear"], dropna=False)["abs_err"].mean().rename("err").reset_index()


def paired(a, b, on):
    m = a.merge(b, on=on, suffixes=("_a", "_b"))
    r = St.paired_contrast(m["err_a"], m["err_b"], m[ID].to_numpy())
    return {k: v for k, v in r.items() if k != "boot"} | {"n_rows": len(m), "n_buildings": int(m[ID].nunique())}


def mae_table(p, keys):
    r = per_row(p, keys)
    out = []
    for k, g in r.groupby(keys, dropna=False):
        est, lo, hi, _ = St.cluster_bootstrap_mean(g["err"], g[ID])
        k = k if isinstance(k, tuple) else (k,)
        out.append(dict(zip(keys, k), mae_log=est, ci_low=lo, ci_high=hi, n_rows=len(g), n_buildings=g[ID].nunique()))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contrasts, sources = {}, {}

    # --- E1: P1, P2 ---------------------------------------------------------------------------------------
    p1, d1 = load_preds("E1")
    sources["E1"] = d1.name
    mae_table(p1, ["split", "tier", "model"]).to_csv(OUT / "table_E1_mae.csv", index=False)
    r1 = per_row(p1, ["split", "tier", "model"])
    for name, tier in (("P1", "T0"), ("P2", "T1")):
        a = r1[(r1.split == "S2_grouped") & (r1.tier == tier) & (r1.model == M4)][[ID, "DataYear", "err"]]
        b = r1[(r1.split == "S1_row_random") & (r1.tier == tier) & (r1.model == M4)][[ID, "DataYear", "err"]]
        contrasts[name] = paired(a, b, [ID, "DataYear"]) | {"definition": f"S2 - S1 | M4 | {tier}"}

    # --- E2: P3 (paired by building: mean error at h=3 vs h=1, over cutoffs) -----------------------------
    p2, d2 = load_preds("E2")
    sources["E2"] = d2.name
    mae_table(p2, ["tier", "model", "horizon"]).to_csv(OUT / "table_E2_mae_by_horizon.csv", index=False)
    mae_table(p2.assign(covid_test=p2["test_year"].isin([2020, 2021])),
              ["tier", "model", "horizon", "covid_test"]).to_csv(OUT / "table_E2_covid_sensitivity.csv", index=False)
    g = p2[(p2.tier == "T0") & (p2.model == M4)].groupby(["horizon", ID])["abs_err"].mean().rename("err").reset_index()
    a = g[g.horizon == 3][[ID, "err"]]
    b = g[g.horizon == 1][[ID, "err"]]
    m = a.merge(b, on=ID, suffixes=("_a", "_b"))
    r = St.paired_contrast(m["err_a"], m["err_b"], m[ID].to_numpy())
    contrasts["P3"] = {k: v for k, v in r.items() if k != "boot"} | {"n_rows": len(m), "n_buildings": len(m),
                                                                     "definition": "S3 h3 - h1 | M4 | T0 (per building)"}

    # --- E3: P4 + E4 heterogeneity --------------------------------------------------------------------------
    p3, d3 = load_preds("E3")
    sources["E3"] = d3.name
    mae_table(p3, ["tier", "model", "horizon"]).to_csv(OUT / "table_E3_mae_by_horizon.csv", index=False)
    mae_table(p3, ["tier", "model"]).to_csv(OUT / "table_E3_mae.csv", index=False)
    r3 = p3[p3.tier == "T0"].groupby(["model", ID, "DataYear", "cutoff"])["abs_err"].mean().rename("err").reset_index()
    a = r3[r3.model == M4][[ID, "DataYear", "cutoff", "err"]]
    b = r3[r3.model == M2][[ID, "DataYear", "cutoff", "err"]]
    contrasts["P4"] = paired(a, b, [ID, "DataYear", "cutoff"]) | {"definition": "M4 - M2 | S4 | T0"}

    h = p3[(p3.tier == "T0") & (p3.model == M4)].copy()
    h["gfa_quintile"] = pd.qcut(h["log_gfa_total"], 5, labels=[f"Q{i}" for i in range(1, 6)]).astype(str)
    h["decade_built"] = (h["YearBuilt"] // 20 * 20).astype(int).astype(str) + "s(20y)"
    het = []
    for col in ["property_type", "gfa_quintile", "decade_built", "Neighborhood"]:
        for lvl, gg in h.groupby(col):
            est, lo, hi, _ = St.cluster_bootstrap_mean(gg["abs_err"], gg[ID])
            het.append({"dimension": col, "level": lvl, "mae_log": est, "ci_low": lo, "ci_high": hi,
                        "n_rows": len(gg), "n_buildings": gg[ID].nunique()})
    pd.DataFrame(het).to_csv(OUT / "table_E4_heterogeneity.csv", index=False)

    # --- E5: P5 --------------------------------------------------------------------------------------------
    p5, d5 = load_preds("E5")
    sources["E5"] = d5.name
    mae_table(p5, ["split", "tier", "model"]).to_csv(OUT / "table_E5_mae_all_rows.csv", index=False)
    mae_table(p5[p5.es_present], ["split", "tier", "model"]).to_csv(OUT / "table_E5_mae_rows_with_es.csv", index=False)
    r5 = per_row(p5[p5.es_present & (p5.split == "S2_grouped") & (p5.model == M4)], ["tier"])
    a = r5[r5.tier == "T0+ES"][[ID, "DataYear", "err"]]
    b = r5[r5.tier == "T0"][[ID, "DataYear", "err"]]
    contrasts["P5"] = paired(a, b, [ID, "DataYear"]) | {"definition": "T0+ES - T0 | S2 | M4 | rows with ES"}

    # --- E6: P6-P8 (already computed per building in the E6 run) --------------------------------------------
    d6 = latest("E6")
    sources["E6"] = d6.name
    c6 = pd.read_csv(d6 / "contrasts.csv")
    for name in ("P6", "P7", "P8"):
        row = c6[(c6.contrast == name) & (c6.model == M4)].iloc[0]
        contrasts[name] = {k: row[k] for k in ("estimate", "ci_low", "ci_high", "p_boot")} | {
            "n_rows": int(row["n_buildings"]), "n_buildings": int(row["n_buildings"]),
            "definition": {"P6": "orig - cons 2016 | GHG | S2 | M4 | T0 common",
                           "P7": "orig - cons 2016 | energy | S2 | M4 | T0 common",
                           "P8": "as-of - snapshot attributes | energy | S2 | M4 | T0 common"}[name]}

    # --- X3 (exploratory, phase 06 review): skill relative to the naive baseline, per vintage cell ------------
    pe = pd.read_csv(d6 / "predictions_e6.csv.gz", dtype={ID: str})
    pe["e"] = (pe["p"] - pe["y"]).abs()
    per = pe.groupby(["target", "attrs", "y_release", "model", ID])["e"].mean().unstack("model")
    x3 = []
    for t in ("ghg", "energy"):
        for a, y in (("orig", "orig"), ("cons", "cons"), ("cons", "orig"), ("orig", "cons")):
            cell = pe[(pe["target"] == t) & (pe["attrs"] == a) & (pe["y_release"] == y)].drop_duplicates(ID)
            x3.append({"target": t, "attrs": a, "y_release": y, "sd_y": cell["y"].std(),
                       "mae_M0": per.loc[(t, a, y), "M0_median_by_type"].mean(),
                       "mae_M4": per.loc[(t, a, y), M4].mean(),
                       "skill_M4_over_M0": per.loc[(t, a, y), M4].mean() / per.loc[(t, a, y), "M0_median_by_type"].mean()})
        o, c = per.loc[(t, "orig", "orig")], per.loc[(t, "cons", "cons")]
        common = o.index.intersection(c.index)
        r = St.paired_contrast(o.loc[common, M4] / o["M0_median_by_type"].mean(),
                               c.loc[common, M4] / c["M0_median_by_type"].mean(), common.to_numpy())
        x3.append({"target": t, "attrs": "contrast", "y_release": "orig-cons skill ratio",
                   "skill_M4_over_M0": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_boot": r["p_boot"]})
    pd.DataFrame(x3).to_csv(OUT / "table_X3_vintage_skill.csv", index=False)

    # --- E6c: the original student result vs protocol v1.0 on the same (original) 2016 release ---------------
    mf = pd.read_csv(d6 / "metrics_by_fold.csv")
    e6c = [{"setting": "Original notebook: one 70/30 random split, top-tail truncation, 1452 buildings",
            "model": "LinearRegression", "r2_log_mean": 0.5345, "r2_log_sd": np.nan, "mae_log_mean": np.nan,
            "source": "results/phase01_reproduction.json (reproduced exactly)"}]
    for mod in ("M2_ridge", "M3_random_forest", M4):
        s = mf[(mf["attrs"] == "orig") & (mf["y_release"] == "orig") & (mf["target"] == "energy") & (mf["model"] == mod)]
        e6c.append({"setting": "Protocol v1.0 on the original 2016 release: grouped 5x5 CV, no truncation, "
                               f"{s['n'].sum() // 5} matched buildings", "model": mod,
                    "r2_log_mean": s["r2_log"].mean(), "r2_log_sd": s["r2_log"].std(),
                    "mae_log_mean": s["mae_log"].mean(), "source": str(d6.relative_to(ROOT))})
    pd.DataFrame(e6c).to_csv(OUT / "table_E6c_original_vs_protocol.csv", index=False)

    adj = St.holm({k: v["p_boot"] for k, v in contrasts.items()})
    rows = []
    for k in sorted(contrasts):
        v = contrasts[k]
        sig = adj[k] < 0.05
        rows.append({"contrast": k, "definition": v["definition"], "estimate_mae_log": v["estimate"],
                     "ci95_low": v["ci_low"], "ci95_high": v["ci_high"], "p_boot": v["p_boot"], "p_holm": adj[k],
                     "significant_holm": sig, "substantial": bool(sig and abs(v["estimate"]) >= THRESHOLD),
                     "n_rows": v["n_rows"], "n_buildings": v["n_buildings"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table_primary_contrasts.csv", index=False)
    (OUT / "sources.json").write_text(json.dumps(sources, indent=2))
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
