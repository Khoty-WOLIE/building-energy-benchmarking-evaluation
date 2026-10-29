"""Phase 01 - faithful re-implementation of the original notebook 1 cleaning pipeline + cheap deterministic models.

Goal: check whether the recorded outputs of WOLIE_Khoty_1_notebook_exploratoire_042024.ipynb can be reproduced
from the raw 2016 file. Each step cites the original cell. Only deterministic / cheap models are refitted
(Dummy, LinearRegression, Ridge grid); no new expensive training is launched.

Usage:
    python src/replicate_original_pipeline.py data/raw/2016_Building_Energy_Benchmarking.csv results/phase01_reproduction.json
"""
import json
import math
import sys

import numpy as np
import pandas as pd
from sklearn import dummy, metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CHECKS = []


def check(name, observed, expected):
    CHECKS.append({"check": name, "observed": observed, "expected_in_notebook": expected,
                   "match": observed == expected})


def clean(raw):
    data = raw.copy()
    data = data.dropna(axis=1, how="all")  # cell 15
    conso = ['SiteEUI(kBtu/sf)', 'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)',
             'SiteEnergyUse(kBtu)', 'SiteEnergyUseWN(kBtu)', 'SteamUse(kBtu)', 'Electricity(kWh)',
             'Electricity(kBtu)', 'NaturalGas(therms)', 'NaturalGas(kBtu)']
    # cell 18 assigns to an unused variable -> no effect. cell 19/21: rows whose summed consumption is 0
    # (this includes the 5 all-NaN rows, since NaN sums to 0).
    tot = data[conso].sum(axis=1).round(2)
    check("cell19_zero_consumption_rows", int((tot == 0).sum()), 12)
    data = data.drop(index=tot[tot == 0].index)
    data = data[~data["BuildingType"].str.contains("Multifamily")].reset_index(drop=True)  # cell 27
    data = data[~data["PrimaryPropertyType"].str.contains("Multifamily|Residence Hall")].reset_index(drop=True)  # cell 29
    data = data.drop(columns=["PropertyName", "Latitude", "Longitude", "Address", "ZipCode", "State", "City",
                              "ListOfAllPropertyUseTypes"])  # cells 33-44 (Coordonnees only used for maps)
    data["Neighborhood"] = data["Neighborhood"].str.upper().str.replace(" NEIGHBORHOODS", "")  # cell 38
    idx = data[data["LargestPropertyUseType"].isna() & (data["NumberofBuildings"] == 1)].index  # cell 48
    for i in idx:
        data.at[i, "LargestPropertyUseTypeGFA"] = data.at[i, "PropertyGFABuilding(s)"]
        data.at[i, "LargestPropertyUseType"] = data.at[i, "PrimaryPropertyType"]
    data = data.dropna(subset=["LargestPropertyUseType"]).reset_index(drop=True)
    del data["YearsENERGYSTARCertified"]  # cell 49
    check("cell51_city_flagged_outliers", int(data["Outlier"].notna().sum()), 16)
    data = data[data["Outlier"].isna()]  # cell 51
    del data["Outlier"]
    check("cell53_default_data_rows", int((data["DefaultData"] == True).sum()), 85)  # noqa: E712
    data = data[data["DefaultData"] == False].reset_index(drop=True)  # cell 53  # noqa: E712
    del data["DefaultData"], data["ComplianceStatus"]
    check("cell56_rows_after_default_filter", int(len(data)), 1535)
    data = data.drop(columns=["DataYear", "OSEBuildingID", "TaxParcelIdentificationNumber"])  # cells 58-63
    data.insert(4, "YearBuiltDec", data["YearBuilt"].apply(lambda v: math.floor(v / 10) * 10))  # cell 67
    data["NumberofBuildings"] = data["NumberofBuildings"].astype(int)  # cell 74

    for col, keys in [("NumberofBuildings", ["Neighborhood", "PropertyGFATotal", "NumberofFloors"]),
                      ("NumberofFloors", ["Neighborhood", "PropertyGFATotal", "NumberofBuildings"])]:  # cells 77/82
        info = data[["Neighborhood", "PropertyGFATotal", "NumberofBuildings", "NumberofFloors"]].copy()
        info["PropertyGFATotal"] = info["PropertyGFATotal"].round(-2)
        means = info.groupby(keys)[col].mean().round().astype(int)
        for i in data[data[col] == 0].index:
            r = data.iloc[i]
            data.loc[i, col] = means[r[keys[0]]][round(r[keys[1]], -2)][r[keys[2]]]
        for i in data[data[col] == 0].index:  # cells 78/83: fallback on neighbourhood mean
            data.loc[i, col] = data.groupby("Neighborhood")[col].mean().round().astype(int)[data.iloc[i]["Neighborhood"]]

    data = data[data["SourceEUIWN(kBtu/sf)"] >= 0].reset_index(drop=True)  # cell 94
    data = data[data["Electricity(kBtu)"] != 0]
    data = data[data["SourceEUI(kBtu/sf)"] != 0].reset_index(drop=True)

    # cell 102: zero WN values replaced by a linear fit on the raw value. Under pandas<2 the chained
    # assignment `data[col].iloc[i] = v` wrote through; under pandas>=3 (Copy-on-Write) it is a silent no-op.
    # Written here with .loc to reproduce the pandas<2 behaviour. These columns are never model inputs.
    for x, y in zip(['SiteEUI(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SiteEnergyUse(kBtu)'],
                    ['SiteEUIWN(kBtu/sf)', 'SourceEUIWN(kBtu/sf)', 'SiteEnergyUseWN(kBtu)']):
        data.loc[(data[y] == 0) & (data[x] != 0), y] = np.nan
        ok = data[data[y].notna()]
        f = np.poly1d(np.polyfit(ok[x], ok[y], deg=1))
        miss = data[y].isna()
        data.loc[miss, y] = f(data.loc[miss, x])

    data = data.drop(columns=['SiteEUI(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SiteEnergyUseWN(kBtu)',
                              'Electricity(kWh)', 'NaturalGas(therms)', 'GHGEmissionsIntensity',
                              'SourceEUIWN(kBtu/sf)'])  # cells 106, 113, 118
    check("cell109_rows_before_target_outliers", int(len(data)), 1520)

    def find_outliers_iqr(s, alpha):  # cell 123, copied logic (upper tail only)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return s[s > (q3 + alpha * (q3 - q1))].index

    def find_all_outliers(df, column):
        good, steps, outliers = 0, 0, []
        for i in range(5, 100):
            tmp = find_outliers_iqr(df[column], i / 5)
            new = round(len(tmp) / len(df) * 100, 1)
            if good != new:
                good = new
            else:
                steps += 1
                if steps == 2:
                    outliers.extend(tmp)
        return list(set(outliers))

    removed = {}
    for t in ["SiteEnergyUse(kBtu)", "TotalGHGEmissions"]:  # cell 124
        out = find_all_outliers(data, t)
        removed[t] = len(out)
        data = data.drop(index=out).reset_index(drop=True)
    check("cell124_outliers_SiteEnergyUse", removed["SiteEnergyUse(kBtu)"], 50)
    check("cell124_outliers_TotalGHGEmissions", removed["TotalGHGEmissions"], 18)
    check("final_rows", int(len(data)), 1452)

    data["Parking (%)"] = data["PropertyGFAParking"] / data["PropertyGFATotal"]  # cell 130
    data = data.drop(columns=['LargestPropertyUseType', 'LargestPropertyUseTypeGFA', 'SecondLargestPropertyUseType',
                              'SecondLargestPropertyUseTypeGFA', 'ThirdLargestPropertyUseType',
                              'ThirdLargestPropertyUseTypeGFA', 'PropertyGFAParking',
                              'CouncilDistrictCode', 'Neighborhood'])  # cells 132, 152
    for c in ["SteamUse(kBtu)", "Electricity(kBtu)", "NaturalGas(kBtu)"]:  # cell 185
        data[c.split("(")[0] + "_binary"] = (data[c] > 0).astype(int)
    data["PropertyGFABuilding (%)"] = data["PropertyGFABuilding(s)"] / data["PropertyGFATotal"]  # cell 186
    data = data.drop(columns=["PropertyGFABuilding(s)"])
    return data


def encode(data, faithful):
    """faithful=True reproduces cells 193-207 (names from .unique(), columns in sorted order)."""
    data = data.copy()
    mapping = {}
    for col in ["BuildingType", "PrimaryPropertyType"]:
        enc = OneHotEncoder()
        lab = enc.fit_transform(data[col].values.reshape(-1, 1)).toarray()
        true_names = [c.lower() for c in enc.categories_[0]]
        names = [c.lower() for c in data[col].unique()] if faithful else true_names
        mapping[col] = {n: t for n, t in zip(names, true_names)}
        data = pd.concat([data, pd.DataFrame(columns=names[:-1], data=lab[:, :-1]).astype(int)], axis=1)
    return data, mapping


FEATURES = ['nonresidential', 'nonresidential cos', 'sps-district k-12', 'campus', 'hotel', 'other',
            'mixed use property', 'university', 'small- and mid-sized office', 'self-storage facility', 'warehouse',
            'k-12 school', 'large office', 'senior care community', 'medical office', 'retail store',
            'distribution center', 'worship facility', 'supermarket / grocery store', 'laboratory',
            'refrigerated warehouse', "YearBuilt", "NumberofBuildings", "NumberofFloors", 'PropertyGFATotal',
            'PropertyGFABuilding (%)', 'Parking (%)', 'SteamUse_binary', 'Electricity_binary', 'NaturalGas_binary']


def fit_eval(data, target, features):
    d = data.copy()
    cols = ["YearBuilt", "NumberofBuildings", "NumberofFloors", "PropertyGFATotal", "PropertyGFABuilding (%)",
            "Parking (%)"]
    d[cols] = StandardScaler().fit_transform(d[cols])  # cell 210: fitted on all rows before the split
    y = np.log1p(d[target])  # cell 211
    X = d[[f for f in features if f in d.columns]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)  # cell 215
    res = {"n_features": X.shape[1], "n_train": len(Xtr), "n_test": len(Xte)}
    models = {"Dummy": dummy.DummyRegressor(strategy="mean"),
              "LinearRegression": LinearRegression(),
              "Ridge(GridSearchCV)": GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 4, 50)}, cv=5)}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p_te, p_tr = m.predict(Xte), m.predict(Xtr)
        res[name] = {"RMSE_test_log": round(float(np.sqrt(metrics.mean_squared_error(yte, p_te))), 4),
                     "R2_test_log": round(float(metrics.r2_score(yte, p_te)), 4),
                     "R2_train_log": round(float(metrics.r2_score(ytr, p_tr)), 4),
                     "MAE_test_kBtu_original_scale": round(float(metrics.mean_absolute_error(np.expm1(yte), np.expm1(p_te))), 1)}
    return res


def main(src, dst):
    raw = pd.read_csv(src)
    check("raw_shape", list(raw.shape), [3376, 46])
    data = clean(raw)
    faithful, mapping = encode(data, faithful=True)
    corrected, _ = encode(data, faithful=False)
    # Same model inputs as the notebook, but each column carries its true category name.
    true_features = FEATURES[:21]
    true_features = [mapping["BuildingType"].get(f, mapping["PrimaryPropertyType"].get(f)) for f in true_features] + FEATURES[21:]
    out = {"checks": CHECKS,
           "label_mapping_notebook_name_to_true_category": mapping,
           "categories_silently_merged_into_reference": {
               "BuildingType": [t for n, t in mapping["BuildingType"].items() if n not in FEATURES],
               "PrimaryPropertyType": [t for n, t in mapping["PrimaryPropertyType"].items() if n not in FEATURES]},
           "results": {}}
    for target in ["SiteEnergyUse(kBtu)", "TotalGHGEmissions"]:
        out["results"][target] = {
            "faithful_encoding": fit_eval(faithful, target, FEATURES),
            "true_labels_same_columns": fit_eval(corrected, target, true_features),
            "corrected_full_onehot": fit_eval(corrected, target, [c for c in corrected.columns if c in
                                              [t.lower() for t in sorted(set(data.BuildingType))[:-1] + sorted(set(data.PrimaryPropertyType))[:-1]]] + FEATURES[21:])}
    out["notebook_recorded_SiteEnergyUse"] = {"Dummy": {"RMSE_test": 1.17, "R2_test": -0.0049},
                                              "LinearRegression": {"RMSE_test": 0.80, "R2_test": 0.5345, "R2_train": 0.6105},
                                              "Ridge": {"RMSE_test": 0.80, "R2_test": 0.5331, "R2_train": 0.6102}}
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
