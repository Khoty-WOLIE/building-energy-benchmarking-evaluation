"""Invariants that the protocol relies on. Run: python -m pytest tests -q (from article-01-seattle/)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from a01 import data as D  # noqa: E402
from a01 import splits as S  # noqa: E402


@pytest.fixture(scope="module")
def toy():
    rng = np.random.RandomState(0)
    rows = []
    for b in range(40):
        for y in range(2015, 2026):
            if rng.rand() < 0.15:
                continue
            e, g, s = rng.uniform(1e5, 1e6), rng.uniform(0, 5e5), 0.0
            rows.append({D.ID: str(b), D.YEAR: y, D.TARGET: e + g + s, "Electricity(kBtu)": e,
                         "NaturalGas(kBtu)": g, "SteamUse(kBtu)": s})
    return pd.DataFrame(rows)


def test_grouped_has_no_building_overlap(toy):
    for sp in S.grouped(toy, k=5, repeats=2):
        assert not set(toy[D.ID].iloc[sp["train"]]) & set(toy[D.ID].iloc[sp["test"]])


def test_row_random_partitions_rows(toy):
    tests = [sp["test"] for sp in S.row_random(toy, k=5, repeats=1)]
    assert sorted(np.concatenate(tests)) == list(range(len(toy)))


def test_forward_temporal_never_trains_on_future(toy):
    for sp in S.forward_temporal(toy):
        assert toy[D.YEAR].iloc[sp["train"]].max() <= sp["cutoff"] < toy[D.YEAR].iloc[sp["test"]].min()


def test_grouped_temporal_is_both(toy):
    for sp in S.grouped_temporal(toy):
        tr, te = toy.iloc[sp["train"]], toy.iloc[sp["test"]]
        assert tr[D.YEAR].max() <= sp["cutoff"] < te[D.YEAR].min()
        assert not set(tr[D.ID]) & set(te[D.ID])


def test_t1_reference_strictly_before_target_and_capped(toy):
    d = D.add_t1_features(toy, max_ref_year=2019)
    ok = d[d["has_ref"]]
    assert (ok["ref_year"] < ok[D.YEAR]).all()
    assert (ok["ref_year"] <= 2019).all()
    shares = ok[["ref_share_electricity", "ref_share_gas", "ref_share_steam"]].sum(axis=1)
    assert np.allclose(shares, 1.0)


def test_first_year_rows_have_no_reference(toy):
    d = D.add_t1_features(toy)
    first = d.sort_values([D.ID, D.YEAR]).groupby(D.ID).head(1)
    assert not first["has_ref"].any()


def test_persistence_uses_only_past(toy):
    d = toy.copy()
    d["y_log"] = np.log(d[D.TARGET])
    p = D.last_prior_log_energy(d, max_year=2018)
    for i in d.index[p.notna()]:
        b, y = d.at[i, D.ID], d.at[i, D.YEAR]
        prior = d[(d[D.ID] == b) & (d[D.YEAR] < y) & (d[D.YEAR] <= 2018)]
        assert np.isclose(p[i], prior.sort_values(D.YEAR)["y_log"].iloc[-1])


def test_tiers_are_disjoint_from_t2():
    t0 = set(D.T0_NUMERIC + D.T0_CATEGORICAL)
    assert not t0 & set(D.T2_FIELDS)
    assert not set(D.T1_EXTRA) & set(D.T2_FIELDS)
