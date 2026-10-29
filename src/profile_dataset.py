"""Phase 01 - descriptive profile of the Seattle Building Energy Benchmarking panel (teqw-tu6e).

No model is trained here. The script only measures what the panel contains, so that the
feasibility of the cross-building / temporal protocols (E1-E3) can be judged on facts.

Usage:
    python src/profile_dataset.py data/raw/seattle_benchmarking_2015_present_teqw-tu6e.csv results/phase01_data_profile.json
"""
import json
import sys

import numpy as np
import pandas as pd

TARGETS = ["SiteEnergyUse(kBtu)", "TotalGHGEmissions"]


def _clean(obj):
    """Make numpy/pandas keys and scalars JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main(src, dst):
    df = pd.read_csv(src, low_memory=False)
    df["DataYear"] = pd.to_numeric(df["DataYear"], errors="coerce").astype("Int64")
    for c in TARGETS + ["ENERGYSTARScore", "PropertyGFATotal", "YearBuilt", "NumberofFloors"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = {"source_file": src, "n_rows": int(len(df)), "n_columns": int(df.shape[1]),
           "columns": list(df.columns)}

    out["rows_per_year"] = df.groupby("DataYear").size().astype(int).to_dict()
    out["duplicate_building_year_pairs"] = int(df.duplicated(["OSEBuildingID", "DataYear"]).sum())
    out["unique_buildings_all"] = int(df["OSEBuildingID"].nunique())
    out["building_type_by_year"] = (
        df.pivot_table(index="BuildingType", columns="DataYear", values="OSEBuildingID",
                       aggfunc="count", fill_value=0).astype(int).to_dict())

    nonres = df[~df["BuildingType"].fillna("").str.contains("Multifamily", case=False)].copy()
    out["nonresidential_rows_per_year"] = nonres.groupby("DataYear").size().astype(int).to_dict()
    out["nonresidential_unique_buildings"] = int(nonres["OSEBuildingID"].nunique())
    years_per_bldg = nonres.groupby("OSEBuildingID")["DataYear"].nunique()
    out["nonresidential_years_per_building_distribution"] = years_per_bldg.value_counts().sort_index().astype(int).to_dict()

    def per_year(frame, fn):
        return {int(y): fn(g) for y, g in frame.groupby("DataYear")}

    out["nonres_energystar_missing_rate_by_year"] = per_year(nonres, lambda g: round(float(g["ENERGYSTARScore"].isna().mean()), 4))
    for t in TARGETS:
        out[f"nonres_{t}_missing_rate_by_year"] = per_year(nonres, lambda g, t=t: round(float(g[t].isna().mean()), 4))
        out[f"nonres_{t}_nonpositive_count_by_year"] = per_year(nonres, lambda g, t=t: int((g[t] <= 0).sum()))
        out[f"nonres_{t}_median_by_year"] = per_year(nonres, lambda g, t=t: float(g[t].median()))
    out["compliance_status_by_year"] = (
        df.pivot_table(index="ComplianceStatus", columns="DataYear", values="OSEBuildingID",
                       aggfunc="count", fill_value=0).astype(int).to_dict())
    out["epa_property_type_n_categories"] = int(nonres["EPAPropertyType"].nunique())

    # Attribute stability for the same building across years (schema / reporting consistency).
    stab = {}
    for c in ["PropertyGFATotal", "YearBuilt", "NumberofFloors", "EPAPropertyType", "BuildingType"]:
        n_vals = nonres.groupby("OSEBuildingID")[c].nunique(dropna=True)
        multi = n_vals[years_per_bldg.reindex(n_vals.index) > 1]
        stab[c] = round(float((multi > 1).mean()), 4)
    out["nonres_share_of_multi_year_buildings_with_changing_attribute"] = stab

    # Year-over-year persistence of log energy for the same building: how much a building "remembers" itself.
    t = "SiteEnergyUse(kBtu)"
    p = nonres[nonres[t] > 0].pivot_table(index="OSEBuildingID", columns="DataYear", values=t, aggfunc="first")
    p = np.log(p)
    yoy = {}
    years = sorted(p.columns)
    for a, b in zip(years[:-1], years[1:]):
        pair = p[[a, b]].dropna()
        if len(pair) > 30:
            yoy[f"{a}->{b}"] = {"n": int(len(pair)), "pearson_r_log": round(float(pair[a].corr(pair[b])), 4)}
    out["nonres_log_site_energy_year_over_year_correlation"] = yoy

    out = _clean(out)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(json.dumps({k: out[k] for k in ["n_rows", "rows_per_year", "nonresidential_rows_per_year",
                                           "nonresidential_unique_buildings",
                                           "nonresidential_years_per_building_distribution",
                                           "duplicate_building_year_pairs",
                                           "nonres_energystar_missing_rate_by_year",
                                           "nonres_share_of_multi_year_buildings_with_changing_attribute",
                                           "nonres_log_site_energy_year_over_year_correlation",
                                           "epa_property_type_n_categories"]}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
