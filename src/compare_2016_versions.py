"""Phase 01 - compare the original 2016 release with the 2016 rows of the consolidated 2015-present release.

Tests whether building attributes in the consolidated panel are year-specific or a later snapshot.

Usage:
    python src/compare_2016_versions.py data/raw/2016_Building_Energy_Benchmarking.csv \
        data/raw/seattle_benchmarking_2015_present_teqw-tu6e.csv results/phase01_version_comparison.json
"""
import json
import sys

import numpy as np
import pandas as pd

PAIRS = {"PropertyGFATotal": "PropertyGFATotal", "NumberofFloors": "NumberofFloors",
         "NumberofBuildings": "NumberofBuildings", "YearBuilt": "YearBuilt",
         "PropertyGFAParking": "PropertyGFAParking", "LargestPropertyUseTypeGFA": "LargestPropertyUseTypeGFA",
         "ENERGYSTARScore": "ENERGYSTARScore", "SiteEnergyUse(kBtu)": "SiteEnergyUse(kBtu)",
         "TotalGHGEmissions": "TotalGHGEmissions", "BuildingType": "BuildingType"}


def main(orig_path, cons_path, dst):
    o = pd.read_csv(orig_path)
    c = pd.read_csv(cons_path, low_memory=False)
    c = c[pd.to_numeric(c["DataYear"], errors="coerce") == 2016]
    o["OSEBuildingID"] = o["OSEBuildingID"].astype(str)
    c["OSEBuildingID"] = c["OSEBuildingID"].astype(str)
    m = o.merge(c, on="OSEBuildingID", suffixes=("_orig", "_cons"))
    out = {"rows_original_2016": int(len(o)), "rows_consolidated_2016": int(len(c)), "matched_on_OSEBuildingID": int(len(m)),
           "only_in_original": int((~o.OSEBuildingID.isin(c.OSEBuildingID)).sum()),
           "only_in_consolidated": int((~c.OSEBuildingID.isin(o.OSEBuildingID)).sum()), "fields": {}}
    for a, b in PAIRS.items():
        x, y = m[f"{a}_orig"], m[f"{b}_cons"]
        both = x.notna() & y.notna()
        if not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
            diff = (x[both].astype(str).str.strip().str.lower() != y[both].astype(str).str.strip().str.lower())
        else:
            xv, yv = pd.to_numeric(x[both]), pd.to_numeric(y[both])
            diff = ~np.isclose(xv, yv, rtol=1e-3, atol=0.5)
        out["fields"][a] = {"n_compared": int(both.sum()), "share_different": round(float(diff.mean()), 4),
                            "missing_orig": int(x.isna().sum()), "missing_cons": int(y.isna().sum())}
    ghg = (m["TotalGHGEmissions_cons"] / m["TotalGHGEmissions_orig"]).replace([np.inf, -np.inf], np.nan).dropna()
    out["TotalGHGEmissions_ratio_consolidated_over_original_quantiles"] = {
        str(k): round(float(v), 3) for k, v in ghg.quantile([.05, .25, .5, .75, .95]).items()}
    gfa = m["PropertyGFATotal_cons"] / m["PropertyGFATotal_orig"]
    gfa = gfa[~np.isclose(gfa, 1, rtol=1e-3)]
    out["PropertyGFATotal_ratio_among_changed_quantiles"] = {
        str(k): round(float(v), 3) for k, v in gfa.quantile([.05, .5, .95]).items()}
    changed = m[m["BuildingType_orig"] != m["BuildingType_cons"]]
    out["BuildingType_transitions"] = {f"{a} -> {b}": int(n) for (a, b), n in
                                       changed.groupby(["BuildingType_orig", "BuildingType_cons"]).size()
                                       .sort_values(ascending=False).items()}
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
