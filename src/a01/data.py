"""Loading, harmonisation, population filters and feature tiers (protocol v1.0, sections 2-3).

Every filter is defined a priori from metadata only; no filter looks at the distribution of the target.
"""
import numpy as np
import pandas as pd

TARGET = "SiteEnergyUse(kBtu)"
TARGET_GHG = "TotalGHGEmissions"
TARGET_WN = "SiteEnergyUseWN(kBtu)"
ID, YEAR = "OSEBuildingID", "DataYear"
MIX = ["Electricity(kBtu)", "NaturalGas(kBtu)", "SteamUse(kBtu)"]
COVID_YEARS = (2020, 2021)

# T0: attributes knowable for a building that has never reported energy.
T0_NUMERIC = ["log_gfa_total", "log_gfa_buildings", "parking_share", "NumberofFloors", "NumberofBuildings",
              "YearBuilt", "Latitude", "Longitude", "largest_use_share"]
T0_CATEGORICAL = ["BuildingType", "property_type", "Neighborhood"]
# T1: T0 + one metered baseline year strictly before the target year (scenario of the original brief).
T1_EXTRA = ["ref_share_electricity", "ref_share_gas", "ref_share_steam", "ref_log_energy", "ref_gap_years"]
# T2: same-year fields computed from the target's own meter data - used ONLY in the leakage analysis (E5).
T2_FIELDS = ["ENERGYSTARScore", "SiteEUI(kBtu/sf)", "SourceEUI(kBtu/sf)", "SiteEUIWN(kBtu/sf)",
             "SourceEUIWN(kBtu/sf)", "GHGEmissionsIntensity"] + MIX

RARE_TYPE_MIN_BUILDINGS = 30


def harmonise_neighborhood(s):
    s = s.fillna("UNKNOWN").str.upper().str.strip()
    s = s.str.replace(" NEIGHBORHOODS", "", regex=False).str.replace(r"\s*/\s*", " / ", regex=True)
    return s


def load_consolidated(path):
    df = pd.read_csv(path, low_memory=False)
    for c in [YEAR, "YearBuilt", "NumberofFloors", "NumberofBuildings", "PropertyGFATotal", "PropertyGFABuildings",
              "PropertyGFAParking", "LargestPropertyUseTypeGFA", "Latitude", "Longitude", TARGET, TARGET_GHG,
              TARGET_WN, "ENERGYSTARScore"] + MIX:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[ID] = df[ID].astype(str)
    df["Neighborhood"] = harmonise_neighborhood(df["Neighborhood"])
    return df


def apply_population(df):
    """Non-residential population + a-priori quality filters. Returns (frame, attrition table)."""
    steps = [("all rows (2015-2025)", pd.Series(True, index=df.index)),
             ("non-residential (BuildingType not Multifamily)", ~df["BuildingType"].fillna("").str.contains("Multifamily")),
             ("ComplianceStatus == Compliant", df["ComplianceStatus"].eq("Compliant")),
             ("not demolished", ~df["Demolished"].fillna(False).astype(bool)),
             ("target present and > 0", df[TARGET].gt(0)),
             ("PropertyGFATotal > 0", df["PropertyGFATotal"].gt(0))]
    keep = pd.Series(True, index=df.index)
    rows = []
    for name, mask in steps:
        keep &= mask
        sub = df[keep]
        rows.append({"step": name, "rows": int(keep.sum()), "buildings": int(sub[ID].nunique()),
                     **{f"y{y}": int((sub[YEAR] == y).sum()) for y in sorted(df[YEAR].dropna().unique().astype(int))}})
    return df[keep].copy(), pd.DataFrame(rows)


def property_type_map(df):
    """Group EPAPropertyType categories with < RARE_TYPE_MIN_BUILDINGS distinct buildings into 'Other (grouped)'.

    Uses category counts only (no target), computed once on the analysis population.
    """
    n_bldg = df.groupby("EPAPropertyType")[ID].nunique()
    keep = set(n_bldg[n_bldg >= RARE_TYPE_MIN_BUILDINGS].index)
    return {t: (t if t in keep else "Other (grouped)") for t in n_bldg.index}


def add_t0_features(df, type_map):
    d = df.copy()
    d["log_gfa_total"] = np.log(d["PropertyGFATotal"])
    d["log_gfa_buildings"] = np.log1p(d["PropertyGFABuildings"].clip(lower=0))
    d["parking_share"] = (d["PropertyGFAParking"].clip(lower=0) / d["PropertyGFATotal"]).clip(0, 1)
    d["largest_use_share"] = (d["LargestPropertyUseTypeGFA"] / d["PropertyGFATotal"]).clip(0, 2)
    d["property_type"] = d["EPAPropertyType"].map(type_map).fillna("Other (grouped)")
    d["y_log"] = np.log(d[TARGET])
    d["is_covid_year"] = d[YEAR].isin(COVID_YEARS)
    return d


def add_t1_features(df, max_ref_year=None, which="first"):
    """Attach the building's FIRST observed year strictly before each row's year as the metered baseline.

    max_ref_year: latest year allowed as reference (training cut-off in temporal splits); None = no cap.
    which="last" (exploratory X2, temporal splits only): use the LAST year <= max_ref_year instead.
    Rows without an admissible reference get NaN T1 features (has_ref = False).
    """
    d = df.copy()
    ref = d[[ID, YEAR, TARGET] + MIX].copy()
    if max_ref_year is not None:
        ref = ref[ref[YEAR] <= max_ref_year]
    ref = ref.sort_values([ID, YEAR])
    pick = ref.groupby(ID).head(1) if which == "first" else ref.groupby(ID).tail(1)
    first = pick.rename(columns={YEAR: "ref_year"})
    tot = first[MIX].fillna(0).sum(axis=1).replace(0, np.nan)
    first["ref_share_electricity"] = first["Electricity(kBtu)"].fillna(0) / tot
    first["ref_share_gas"] = first["NaturalGas(kBtu)"].fillna(0) / tot
    first["ref_share_steam"] = first["SteamUse(kBtu)"].fillna(0) / tot
    first["ref_log_energy"] = np.log(first[TARGET])
    d = d.merge(first[[ID, "ref_year", "ref_share_electricity", "ref_share_gas", "ref_share_steam", "ref_log_energy"]],
                on=ID, how="left")
    admissible = d["ref_year"] < d[YEAR]
    for c in ["ref_year"] + T1_EXTRA[:-1]:
        d.loc[~admissible, c] = np.nan
    d["ref_gap_years"] = d[YEAR] - d["ref_year"]
    d["has_ref"] = admissible
    return d


def last_prior_log_energy(df, max_year=None):
    """Persistence baseline input: log energy of the most recent prior year (<= max_year if given)."""
    d = df[[ID, YEAR, "y_log"]].sort_values([ID, YEAR])
    out = pd.Series(np.nan, index=df.index)
    by = {k: g for k, g in d.groupby(ID)}
    for i, (b, y) in df[[ID, YEAR]].iterrows():
        g = by[b]
        cap = y - 1 if max_year is None else min(y - 1, max_year)
        prev = g[g[YEAR] <= cap]
        if len(prev):
            out[i] = prev["y_log"].iloc[-1]
    return out
