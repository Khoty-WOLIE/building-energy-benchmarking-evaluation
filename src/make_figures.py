"""Phase 04 figures (PDF + PNG) from results/phase04 tables and the latest E6 run.

Palette: validated categorical slots 1-3 (light mode), text in neutral ink, one axis per chart,
CIs as thin whiskers, direct labels instead of legends where <= 3 series.

Usage (from article-01-seattle/):
    python src/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "phase04"
FIG = ROOT / "manuscript" / "figures"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
M4 = "M4_hist_gradient_boosting"
MODEL_LABEL = {"M0_median_by_type": "Median by type", "M1_persistence": "Persistence",
               "M2_ridge": "Ridge", "M3_random_forest": "Random forest", M4: "Gradient boosting"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def dot_ci(ax, labels, est, lo, hi, color, offset=0.0, label=None):
    ys = [i + offset for i in range(len(labels))]
    ax.errorbar(est, ys, xerr=[[e - l for e, l in zip(est, lo)], [h - e for e, h in zip(est, hi)]], fmt="o",
                color=color, ms=6, lw=1.2, capsize=0, label=label, markeredgecolor="white", markeredgewidth=1)
    return ys


def fig_e6():
    run = sorted((ROOT / "results" / "experiments" / "E6").glob("*"))[-1]
    s = pd.read_csv(run / "summary_mae.csv")
    s = s[s.model == M4]
    cells = [("orig", "orig"), ("orig", "cons"), ("cons", "orig"), ("cons", "cons")]
    names = ["Original attributes, original target", "Original attributes, consolidated target",
             "Consolidated attributes, original target", "Consolidated attributes, consolidated target"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), sharey=True)
    for ax, (t, title) in zip(axes, [("energy", "Site energy"), ("ghg", "GHG emissions")]):
        g = s[s["target"] == t].set_index(["attrs", "y_release"]).loc[cells]
        dot_ci(ax, names, g["mae_log"], g["ci_low"], g["ci_high"], SLOT[0])
        ax.set_title(title, color=INK, fontsize=9, loc="left")
    axes[0].set_yticks(range(4), names)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    fig.supxlabel("MAE (log scale), gradient boosting, building-grouped CV, 2016 matched buildings",
                  fontsize=9, color=INK, y=-0.04)
    save(fig, "fig_e6_vintage_factorial")


def fig_e1():
    t = pd.read_csv(TAB / "table_E1_mae.csv")
    t = t[t.tier == "T0"]
    models = ["M0_median_by_type", "M2_ridge", "M3_random_forest", M4]
    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    for i, (sp, lab) in enumerate([("S1_row_random", "Row-random CV"), ("S2_grouped", "Building-grouped CV")]):
        g = t[t.split == sp].set_index("model").loc[models]
        ys = dot_ci(ax, models, g["mae_log"], g["ci_low"], g["ci_high"], SLOT[i], offset=-0.12 + 0.24 * i)
        ax.annotate(lab, (g["ci_high"].iloc[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models])
    ax.invert_yaxis()
    ax.set_xlabel("MAE (log scale), tier T0")
    save(fig, "fig_e1_random_vs_grouped")


def fig_horizon():
    e2 = pd.read_csv(TAB / "table_E2_mae_by_horizon.csv")
    e3 = pd.read_csv(TAB / "table_E3_mae_by_horizon.csv")
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for i, (tab, lab) in enumerate([(e2, "Forward temporal (S3)"), (e3, "Unseen buildings + future (S4)")]):
        g = tab[(tab.tier == "T0") & (tab.model == M4)].sort_values("horizon")
        ax.plot(g["horizon"], g["mae_log"], color=SLOT[i], lw=2, marker="o", ms=6, markeredgecolor="white")
        ax.fill_between(g["horizon"], g["ci_low"], g["ci_high"], color=SLOT[i], alpha=0.15, lw=0)
        ax.annotate(lab, (g["horizon"].iloc[-1], g["mae_log"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Horizon (years after training cut-off)")
    ax.set_ylabel("MAE (log scale)")
    save(fig, "fig_e2_e3_horizon")


def fig_e5():
    t = pd.read_csv(TAB / "table_E5_mae_rows_with_es.csv")
    t = t[(t.model == M4) & (t.split == "S2_grouped")]
    tiers = ["T0", "T0+ES", "T1", "T1+ES", "T0+EUI"]
    t = t.set_index("tier").loc[[x for x in tiers if x in set(t.tier)]]
    nice = {"T0": "T0 structural", "T0+ES": "T0 + ENERGY STAR", "T1": "T1 (T0 + baseline year)",
            "T1+ES": "T1 + ENERGY STAR", "T0+EUI": "T0 + same-year EUI (ceiling)"}
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    dot_ci(ax, list(t.index), t["mae_log"], t["ci_low"], t["ci_high"], SLOT[0])
    ax.set_yticks(range(len(t)), [nice[x] for x in t.index])
    ax.invert_yaxis()
    ax.set_xlabel("MAE (log scale), gradient boosting, grouped CV, rows with ENERGY STAR")
    save(fig, "fig_e5_energy_star_leakage")


def fig_e4():
    t = pd.read_csv(TAB / "table_E4_heterogeneity.csv")
    g = t[t.dimension == "property_type"].sort_values("mae_log")
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(g) + 0.8))
    dot_ci(ax, list(g["level"]), g["mae_log"], g["ci_low"], g["ci_high"], SLOT[0])
    ax.set_yticks(range(len(g)), [f"{lv} (n={n})" for lv, n in zip(g["level"], g["n_buildings"])])
    ax.set_xlabel("MAE (log scale), gradient boosting, S4, tier T0")
    save(fig, "fig_e4_heterogeneity_property_type")


if __name__ == "__main__":
    for f in (fig_e6, fig_e1, fig_horizon, fig_e5, fig_e4):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
