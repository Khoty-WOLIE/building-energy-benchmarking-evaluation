"""Phase 03 - apply the protocol's data definitions to the frozen snapshot and summarise what the experiments will see.

No model is trained. Outputs (results/): phase03_attrition.csv, phase03_property_type_map.csv,
phase03_tier_availability.csv, phase03_split_summary.csv, phase03_summary.json

Usage (from article-01-seattle/):
    python src/prepare_phase03.py data/raw/seattle_benchmarking_2015_present_teqw-tu6e.csv \
        data/raw/2016_Building_Energy_Benchmarking.csv results
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a01 import data as D  # noqa: E402
from a01 import splits as S  # noqa: E402

EXPECTED_SHA256 = "d1a8227bf9c7c82aea8279a67405add617271852a985d96bb6e4ac9070e8250f"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(cons_path, orig2016_path, out_dir):
    out = Path(out_dir)
    digest = sha256(cons_path)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"Snapshot mismatch: {digest} != frozen {EXPECTED_SHA256}")

    raw = D.load_consolidated(cons_path)
    pop, attrition = D.apply_population(raw)
    attrition.to_csv(out / "phase03_attrition.csv", index=False)

    tmap = D.property_type_map(pop)
    pd.Series(tmap, name="property_type").rename_axis("EPAPropertyType").to_csv(out / "phase03_property_type_map.csv")
    d = D.add_t1_features(D.add_t0_features(pop, tmap))

    avail = []
    for tier, cols in [("T0", D.T0_NUMERIC + D.T0_CATEGORICAL), ("T1_extra", D.T1_EXTRA), ("T2", D.T2_FIELDS)]:
        for c in cols:
            avail.append({"tier": tier, "feature": c, "missing_rate": round(float(d[c].isna().mean()), 4)})
    pd.DataFrame(avail).to_csv(out / "phase03_tier_availability.csv", index=False)

    rows = []
    for gen in (S.row_random(d, repeats=1), S.grouped(d, repeats=1), S.forward_temporal(d), S.grouped_temporal(d)):
        for sp in gen:
            te = d.iloc[sp["test"]]
            rows.append({k: v for k, v in sp.items() if k not in ("train", "test")} |
                        {"n_train": len(sp["train"]), "n_test": len(sp["test"]),
                         "test_buildings": int(te[D.ID].nunique()),
                         "test_share_with_t1_ref": round(float(te["has_ref"].mean()), 3)})
    split_summary = pd.DataFrame(rows)
    split_summary.to_csv(out / "phase03_split_summary.csv", index=False)

    years_per_b = d.groupby(D.ID)[D.YEAR].nunique()
    multi = years_per_b[years_per_b > 1].index
    static = {c: round(float((d[d[D.ID].isin(multi)].groupby(D.ID)[c].nunique() > 1).mean()), 4)
              for c in ["PropertyGFATotal", "NumberofFloors", "YearBuilt", "property_type", "LargestPropertyUseTypeGFA"]}

    o = pd.read_csv(orig2016_path, low_memory=False)
    o[D.ID] = o[D.ID].astype(str)
    matched_2016 = int(d[(d[D.YEAR] == 2016) & d[D.ID].isin(o[D.ID])][D.ID].nunique())

    summary = {
        "snapshot_sha256": digest,
        "analysis_rows": int(len(d)), "analysis_buildings": int(d[D.ID].nunique()),
        "rows_per_year": {int(k): int(v) for k, v in d.groupby(D.YEAR).size().items()},
        "property_types_kept": int(len({v for v in tmap.values()})),
        "rows_with_t1_reference": int(d["has_ref"].sum()),
        "share_rows_with_t1_reference": round(float(d["has_ref"].mean()), 4),
        "energy_star_missing_rate": round(float(d["ENERGYSTARScore"].isna().mean()), 4),
        "share_multi_year_buildings_with_changing_attribute": static,
        "buildings_2016_matched_to_original_release": matched_2016,
        "n_split_instances": split_summary.groupby("split").size().to_dict(),
        "s3_test_rows_by_horizon": {int(h): int(n) for h, n in split_summary[split_summary.split == "S3_forward_temporal"]
                                    .groupby("horizon")["n_test"].sum().items()},
    }
    with open(out / "phase03_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=int)
    print(attrition.to_string(index=False))
    print(json.dumps(summary, indent=1, default=int))


if __name__ == "__main__":
    main(*sys.argv[1:4])
