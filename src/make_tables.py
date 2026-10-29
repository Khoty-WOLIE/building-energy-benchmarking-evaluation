"""Generate LaTeX tables for the manuscript directly from result files (no manual transcription).

Usage (from article-01-seattle/):
    python src/make_tables.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = RES / "phase04"
OUT = ROOT / "manuscript" / "tables"
M4, M2, M3 = "M4_hist_gradient_boosting", "M2_ridge", "M3_random_forest"
MODEL = {"M0_median_by_type": "Median by type", "M1_persistence": "Persistence (last year)", M2: "Ridge",
         M3: "Random forest", M4: "Gradient boosting"}


def f3(x):
    return f"{x:.3f}"


def s3(x):
    """Signed number typeset with a true plus or minus sign (never a text hyphen)."""
    return f"${x:+.3f}$"


def ci(r, est="mae_log", lo="ci_low", hi="ci_high"):
    return f"{r[est]:.3f} [{r[lo]:.3f}, {r[hi]:.3f}]"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print("wrote", name)


def latest(exp):
    return sorted(p for p in (RES / "experiments" / exp).glob("*") if p.is_dir())[-1]


def t_vintage():
    v = json.loads((RES / "phase01_version_comparison.json").read_text())
    labels = {"TotalGHGEmissions": "Total GHG emissions", "PropertyGFAParking": "Parking floor area",
              "BuildingType": "Building type", "PropertyGFATotal": "Total floor area",
              "NumberofBuildings": "Number of buildings", "ENERGYSTARScore": "ENERGY STAR score",
              "SiteEnergyUse(kBtu)": "Site energy use", "LargestPropertyUseTypeGFA": "Largest-use floor area",
              "NumberofFloors": "Number of floors", "YearBuilt": "Year built"}
    rows = sorted(v["fields"].items(), key=lambda kv: -kv[1]["share_different"])
    body = "\n".join(f"{labels[k]} & {d['n_compared']:,} & {100 * d['share_different']:.1f}\\% \\\\" for k, d in rows)
    q = v["TotalGHGEmissions_ratio_consolidated_over_original_quantiles"]
    write("tab_vintage", rf"""\begin{{tabular}}{{lrr}}
\toprule
Field (reporting year 2016) & Buildings compared & Share revised \\
\midrule
{body}
\bottomrule
\end{{tabular}}
% GHG ratio consolidated/original quantiles: p05={q['0.05']}, p25={q['0.25']}, median={q['0.5']}, p95={q['0.95']}
""")


def t_contrasts():
    c = pd.read_csv(TAB / "table_primary_contrasts.csv")
    desc = {"P1": "Building-grouped minus row-random CV (T0)", "P2": "Building-grouped minus row-random CV (T1)",
            "P3": "Forward temporal, horizon 3 minus horizon 1 (T0)",
            "P4": "Gradient boosting minus Ridge, unseen buildings and future years (T0)",
            "P5": "T0 with ENERGY STAR minus T0, grouped CV", "P6": "Original minus consolidated 2016 release, GHG target",
            "P7": "Original minus consolidated 2016 release, energy target",
            "P8": "As-of minus snapshot attributes, energy target"}
    lines = []
    for _, r in c.iterrows():
        verdict = "substantial" if r["substantial"] else ("significant, below threshold" if r["significant_holm"] else "not significant")
        p = "$<0.005$" if r["p_holm"] < 0.005 else f"{r['p_holm']:.3f}"
        lines.append(f"{r['contrast']} & {desc[r['contrast']]} & {s3(r['estimate_mae_log'])} & "
                     f"[{s3(r['ci95_low'])}, {s3(r['ci95_high'])}] & {p} & {verdict} \\\\")
    write("tab_contrasts", r"""\begin{tabular}{llrcrl}
\toprule
 & Contrast (difference in MAE on the log scale) & Estimate & 95\% CI & $p_{\mathrm{Holm}}$ & Verdict \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def t_e1():
    t = pd.read_csv(TAB / "table_E1_mae.csv")
    models = ["M0_median_by_type", M2, M3, M4]
    lines = []
    for m in models:
        cells = []
        for tier in ("T0", "T1"):
            for sp in ("S1_row_random", "S2_grouped"):
                r = t[(t["model"] == m) & (t["tier"] == tier) & (t["split"] == sp)].iloc[0]
                cells.append(f3(r["mae_log"]))
        lines.append(f"{MODEL[m]} & " + " & ".join(cells) + r" \\")
    write("tab_e1", r"""\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{T0 (15{,}621 rows)} & \multicolumn{2}{c}{T1 (13{,}954 rows)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & Row-random & Grouped & Row-random & Grouped \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def t_horizon():
    e2 = pd.read_csv(TAB / "table_E2_mae_by_horizon.csv")
    e3 = pd.read_csv(TAB / "table_E3_mae_by_horizon.csv")
    lines = []
    for tier, models in (("T0", ["M0_median_by_type", M2, M3, M4]), ("T1", ["M1_persistence", M2, M3, M4])):
        for m in models:
            cells = []
            for tab in (e2, e3):
                for h in (1, 2, 3):
                    r = tab[(tab["tier"] == tier) & (tab["model"] == m) & (tab["horizon"] == h)].iloc[0]
                    cells.append(f3(r["mae_log"]))
            lines.append(f"{tier} & {MODEL[m]} & " + " & ".join(cells) + r" \\")
        if tier == "T0":
            lines.append(r"\midrule")
    write("tab_horizon", r"""\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{Forward temporal (S3)} & \multicolumn{3}{c}{Unseen buildings + future (S4)} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Tier & Model & $h=1$ & $h=2$ & $h=3$ & $h=1$ & $h=2$ & $h=3$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def t_e5():
    t = pd.read_csv(TAB / "table_E5_mae_rows_with_es.csv")
    tiers = [("T0", "T0 (structural)"), ("T0+ES", "T0 + ENERGY STAR"), ("T1", "T1 (T0 + baseline year)"),
             ("T1+ES", "T1 + ENERGY STAR"), ("T0+EUI", "T0 + same-year EUI (leakage ceiling)")]
    lines = []
    for k, lab in tiers:
        cells = []
        for sp in ("S2_grouped", "S4_grouped_temporal"):
            for m in (M2, M4):
                r = t[(t["tier"] == k) & (t["split"] == sp) & (t["model"] == m)].iloc[0]
                cells.append(f3(r["mae_log"]))
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    write("tab_e5", r"""\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Grouped CV (S2)} & \multicolumn{2}{c}{Unseen + future (S4)} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Information tier & Ridge & Gr.\ boosting & Ridge & Gr.\ boosting \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def t_e6():
    s = pd.read_csv(latest("E6") / "summary_mae.csv")
    cells = [("orig", "orig"), ("orig", "cons"), ("cons", "orig"), ("cons", "cons")]
    name = {"orig": "original", "cons": "consolidated"}
    lines = []
    for a, y in cells:
        row = [f"{name[a]} & {name[y]}"]
        for t in ("energy", "ghg"):
            for m in (M2, M4):
                r = s[(s["target"] == t) & (s["attrs"] == a) & (s["y_release"] == y) & (s["model"] == m)].iloc[0]
                row.append(f3(r["mae_log"]))
        lines.append(" & ".join(row) + r" \\")
    write("tab_e6", r"""\begin{tabular}{llcccc}
\toprule
 & & \multicolumn{2}{c}{Site energy} & \multicolumn{2}{c}{GHG emissions} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Attributes & Target & Ridge & Gr.\ boosting & Ridge & Gr.\ boosting \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def t_attrition():
    a = pd.read_csv(RES / "phase03_attrition.csv")
    label = {"all rows (2015-2025)": "All records, 2015 to 2025",
             "non-residential (BuildingType not Multifamily)": "Non-residential buildings",
             "ComplianceStatus == Compliant": "Compliant reports",
             "not demolished": "Not demolished",
             "target present and > 0": "Site energy reported and positive",
             "PropertyGFATotal > 0": "Total floor area positive"}
    lines = [f"{label[r['step']]} & {r['rows']:,} & {r['buildings']:,} \\\\" for _, r in a.iterrows()]
    write("tab_attrition", r"""\begin{tabular}{lrr}
\toprule
Selection step (cumulative) & Building-years & Buildings \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def t_exploratory():
    try:
        x1 = pd.read_csv(latest("X1") / "summary_X1.csv")
        x2 = pd.read_csv(latest("X2") / "summary_X2.csv")
    except (IndexError, FileNotFoundError):
        print("skip exploratory tables")
        return
    lines = []
    for m in (M2, M3, M4):
        cells = []
        for v in ("exact", "jittered"):
            for sp in ("S1_row_random", "S2_grouped"):
                r = x1[(x1["variant"] == v) & (x1["split"] == sp) & (x1["model"] == m)].iloc[0]
                cells.append(f3(r["mae_log"]))
        lines.append(f"{MODEL[m]} & " + " & ".join(cells) + r" \\")
    write("tab_x1", r"""\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{Exact attributes} & \multicolumn{2}{c}{Jittered attributes} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & Row-random & Grouped & Row-random & Grouped \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")
    lines = []
    for m in ("M1_persistence", M2, M4):
        cells = [f3(x2[(x2["model"] == m) & (x2["horizon"].astype(str) == str(h))].iloc[0]["mae_log"]) for h in (1, 2, 3, "all")]
        lines.append(f"{MODEL[m]} & " + " & ".join(cells) + r" \\")
    write("tab_x2", r"""\begin{tabular}{lcccc}
\toprule
Model (T1 with last-year baseline, S4) & $h=1$ & $h=2$ & $h=3$ & All \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def t_x3():
    x = pd.read_csv(TAB / "table_X3_vintage_skill.csv")
    name = {"orig": "original", "cons": "consolidated"}
    lines = []
    for t, lab in (("energy", "Site energy"), ("ghg", "GHG emissions")):
        for a, y in (("orig", "orig"), ("cons", "cons")):
            r = x[(x["target"] == t) & (x["attrs"] == a) & (x["y_release"] == y)].iloc[0]
            lines.append(f"{lab} & {name[a]} & {r['sd_y']:.3f} & {r['mae_M0']:.3f} & {r['mae_M4']:.3f} & "
                         f"{r['skill_M4_over_M0']:.3f} \\\\")
        c = x[(x["target"] == t) & (x["attrs"] == "contrast")].iloc[0]
        lines.append(f"\\multicolumn{{5}}{{r}}{{Paired difference, original minus consolidated [95\\% CI]}} & "
                     f"{s3(c['skill_M4_over_M0'])} [{s3(c['ci_low'])}, {s3(c['ci_high'])}] \\\\")
        if t == "energy":
            lines.append(r"\midrule")
    write("tab_x3", r"""\begin{tabular}{llrrrr}
\toprule
Target & Release & SD of $\log y$ & MAE naive & MAE boosting & Ratio \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


if __name__ == "__main__":
    for fn in (t_vintage, t_contrasts, t_e1, t_horizon, t_e5, t_e6, t_attrition, t_exploratory, t_x3):
        fn()
