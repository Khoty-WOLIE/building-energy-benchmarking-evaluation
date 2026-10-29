"""Static checks of manuscript/main.tex: citations, labels, inputs, figures, braces, and every number quoted in the
text recomputed from the result files. Exit code 1 on any failure.

Usage (from article-01-seattle/):
    python src/check_manuscript.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
RES = ROOT / "results"
TAB = RES / "phase04"
M2, M3, M4 = "M2_ridge", "M3_random_forest", "M4_hist_gradient_boosting"


def latest(exp):
    return sorted(p for p in (RES / "experiments" / exp).glob("*") if p.is_dir())[-1]


def q(df, **kw):
    for k, v in kw.items():
        df = df[df[k].astype(str) == str(v)]
    assert len(df) == 1, (kw, len(df))
    return df.iloc[0]


def expected_numbers():
    e1 = pd.read_csv(TAB / "table_E1_mae.csv")
    e2 = pd.read_csv(TAB / "table_E2_mae_by_horizon.csv")
    e3 = pd.read_csv(TAB / "table_E3_mae_by_horizon.csv")
    e3o = pd.read_csv(TAB / "table_E3_mae.csv")
    cov = pd.read_csv(TAB / "table_E2_covid_sensitivity.csv")
    e5 = pd.read_csv(TAB / "table_E5_mae_rows_with_es.csv")
    e4 = pd.read_csv(TAB / "table_E4_heterogeneity.csv")
    c = pd.read_csv(TAB / "table_primary_contrasts.csv").set_index("contrast")
    c6 = pd.read_csv(latest("E6") / "contrasts.csv")
    e6c = pd.read_csv(TAB / "table_E6c_original_vs_protocol.csv")
    v = json.loads((RES / "phase01_version_comparison.json").read_text())
    prof = json.loads((RES / "phase03_summary.json").read_text())
    f = lambda x: f"{x:.3f}"  # noqa: E731
    exp = {
        "RF S1 T0": f(q(e1, split="S1_row_random", tier="T0", model=M3).mae_log),
        "RF S2 T0": f(q(e1, split="S2_grouped", tier="T0", model=M3).mae_log),
        "GB S1 T0": f(q(e1, split="S1_row_random", tier="T0", model=M4).mae_log),
        "GB S2 T0": f(q(e1, split="S2_grouped", tier="T0", model=M4).mae_log),
        "Ridge S1 T0": f(q(e1, split="S1_row_random", tier="T0", model=M2).mae_log),
        "Ridge S2 T0": f(q(e1, split="S2_grouped", tier="T0", model=M2).mae_log),
        "P1": f"{c.loc['P1', 'estimate_mae_log']:+.3f}", "P2": f"{c.loc['P2', 'estimate_mae_log']:+.3f}",
        "P3": f"{c.loc['P3', 'estimate_mae_log']:+.3f}", "P4": f"{c.loc['P4', 'estimate_mae_log']:+.3f}",
        "P5": f"{c.loc['P5', 'estimate_mae_log']:+.3f}".replace("+", ""),
        "P6": f"{c.loc['P6', 'estimate_mae_log']:+.3f}".replace("+", ""),
        "P6 CI": f"[{-c.loc['P6', 'ci95_high']:.3f}, {-c.loc['P6', 'ci95_low']:.3f}]",
        "S3 h1": f(q(e2, tier="T0", model=M4, horizon=1).mae_log), "S3 h3": f(q(e2, tier="T0", model=M4, horizon=3).mae_log),
        "S4 h1": f(q(e3, tier="T0", model=M4, horizon=1).mae_log), "S4 h3": f(q(e3, tier="T0", model=M4, horizon=3).mae_log),
        "persist h1": f(q(e3, tier="T1", model="M1_persistence", horizon=1).mae_log),
        "Ridge S4 T0": f(q(e3o, tier="T0", model=M2).mae_log), "Ridge S4 T1": f(q(e3o, tier="T1", model=M2).mae_log),
        "covid": f(q(cov, tier="T0", model=M4, horizon=1, covid_test=True).mae_log),
        "noncovid": f(q(cov, tier="T0", model=M4, horizon=1, covid_test=False).mae_log),
        "E5 T0": f(q(e5, split="S2_grouped", tier="T0", model=M4).mae_log),
        "E5 T0+ES": f(q(e5, split="S2_grouped", tier="T0+ES", model=M4).mae_log),
        "E5 T1": f(q(e5, split="S2_grouped", tier="T1", model=M4).mae_log),
        "E5 T1+ES": f(q(e5, split="S2_grouped", tier="T1+ES", model=M4).mae_log),
        "E5 EUI": f(q(e5, split="S2_grouped", tier="T0+EUI", model=M4).mae_log),
        "K12": f(q(e4, dimension="property_type", level="K-12 School").mae_log),
        "Q1": f(q(e4, dimension="gfa_quintile", level="Q1").mae_log),
        "Q5": f(q(e4, dimension="gfa_quintile", level="Q5").mae_log),
        "Duwamish": f(q(e4, dimension="Neighborhood", level="GREATER DUWAMISH").mae_log),
        "target effect": f"{q(c6, contrast='X_target_effect_ghg', model=M4).estimate:.3f}",
        "attr effect": f"{q(c6, contrast='X_attr_effect_ghg', model=M4).estimate:+.3f}".replace("+", "+"),
        "E6c ridge R2": f"{q(e6c, model=M2).r2_log_mean:.3f}",
        "GHG revised": f"{100 * v['fields']['TotalGHGEmissions']['share_different']:.1f}",
        "GHG ratio median": f"{v['TotalGHGEmissions_ratio_consolidated_over_original_quantiles']['0.5']:.3f}",
        "rows": f"{prof['analysis_rows']:,}".replace(",", "{,}"),
        "buildings": f"{prof['analysis_buildings']:,}".replace(",", "{,}"),
    }
    x3 = pd.read_csv(TAB / "table_X3_vintage_skill.csv")
    for rel, lab in (("orig", "orig"), ("cons", "cons")):
        r = q(x3, target="ghg", attrs=rel, y_release=rel)
        exp[f"X3 sd {lab}"] = f(r.sd_y)
        exp[f"X3 skill {lab}"] = f(r.skill_M4_over_M0)
    r = q(x3, target="ghg", attrs="contrast")
    exp["X3 diff"] = f"{r.skill_M4_over_M0:.3f}"
    exp["X3 CI"] = f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"
    e1b = pd.read_csv(TAB / "table_E1_mae.csv")
    exp["GB factor"] = f"{q(e1b, split='S2_grouped', tier='T0', model=M4).mae_log / q(e1b, split='S1_row_random', tier='T0', model=M4).mae_log:.1f}"
    e5b = pd.read_csv(TAB / "table_E5_mae_rows_with_es.csv")
    exp["ES share"] = f"{100 * q(e5b, split='S2_grouped', tier='T0', model=M4).n_rows / prof['analysis_rows']:.0f}\\%"
    x2 = pd.read_csv(latest("X2") / "summary_X2.csv")
    for m, lab in (("M1_persistence", "persist"), (M2, "ridge"), (M4, "gb")):
        exp[f"X2 {lab} all"] = f(q(x2, model=m, horizon="all").mae_log)
        exp[f"X2 {lab} h1"] = f(q(x2, model=m, horizon=1).mae_log)
    x1_dir = sorted((RES / "experiments" / "X1").glob("*/summary_X1.csv"))
    if x1_dir:
        x1 = pd.read_csv(x1_dir[-1])
        for m, lab in ((M3, "rf"), (M4, "gb")):           # values quoted in the text (the rest is in tab_x1)
            for v, s in (("exact", "S1_row_random"), ("jittered", "S1_row_random"), ("jittered", "S2_grouped")):
                exp[f"X1 {lab} {v} {s}"] = f(q(x1, variant=v, split=s, model=m).mae_log)
    return exp


def main():
    tex_file = sys.argv[1] if len(sys.argv) > 1 else "main.tex"   # e.g. main_fr.tex for the French reading copy
    tex = (MS / tex_file).read_text(encoding="utf-8")
    bib = (ROOT / "literature" / "references.bib").read_text(encoding="utf-8")
    errors = []
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    for group in re.findall(r"\\cite[pt]?\{([^}]*)\}", tex):
        for k in group.split(","):
            if k.strip() not in keys:
                errors.append(f"missing bib key: {k.strip()}")
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    for r in re.findall(r"\\ref\{([^}]*)\}", tex):
        if r not in labels:
            errors.append(f"undefined ref: {r}")
    for i in re.findall(r"\\input\{([^}]*)\}", tex):
        if not (MS / f"{i}.tex").exists():
            errors.append(f"missing input: {i}")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        if not (MS / "figures" / g).exists():
            errors.append(f"missing figure: {g}")
    body = re.sub(r"\\[{}]", "", tex)
    if body.count("{") != body.count("}"):
        errors.append(f"unbalanced braces: {body.count('{')} vs {body.count('}')}")
    for env in set(re.findall(r"\\begin\{(\w+)\}", tex)):
        if tex.count(f"\\begin{{{env}}}") != tex.count(f"\\end{{{env}}}"):
            errors.append(f"unbalanced environment: {env}")
    for name, value in expected_numbers().items():
        if value not in tex:
            errors.append(f"number not found in text: {name} = {value}")
    # House style requested by the author: no em dashes, no colons in running text, no stock AI phrasing.
    prose = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith(("\\usepackage", "\\captionsetup",
                                                                                  "\\graphicspath", "%")))
    prose = re.sub(r"\\(label|ref|cite[pt]?|input|includegraphics|url|href)\{[^}]*\}", "", prose)
    for path in [MS / tex_file] + sorted((MS / "tables").glob("*.tex")):
        txt = path.read_text(encoding="utf-8")
        if "---" in txt or "\u2014" in txt:
            errors.append(f"em dash found in {path.name}")
        if re.search(r"(?<![$\w])-\d", re.sub(r"\$[^$]*\$", "", txt)):
            errors.append(f"negative number typeset with a text hyphen in {path.name}")
        if re.search(r"[A-Za-z)] \$-\$ [A-Za-z(]", txt):
            errors.append(f"minus sign between words in {path.name} (write 'minus')")
    if re.search(r"[A-Za-z)]:\s", re.sub(r"\$[^$]*\$", "", prose)):
        errors.append("colon found in running text")
    banned = r"\b(delve|crucial|notably|moreover|furthermore|leverag\w*|landscape|underscore\w*|pivotal|seamless\w*)\b"
    for w in re.findall(banned, prose, flags=re.I):
        errors.append(f"style: avoid '{w}'")
    placeholders = re.findall(r"\[X[12] results to be inserted\.\]", tex)
    if placeholders:
        errors.append(f"{len(placeholders)} placeholder(s) left")
    cited = sorted({k.strip() for g in re.findall(r"\\cite[pt]?\{([^}]*)\}", tex) for k in g.split(",")})
    print(f"citations used: {len(cited)} / bib entries: {len(keys)}")
    print("\n".join(errors) if errors else "ALL CHECKS PASSED")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
