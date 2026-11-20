"""The mesh block build: counts, the rented estimate, the overlay, weighting."""

import dataclasses

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from abs_methods import build_weightings as bw
from abs_methods import mesh_block_weightings as mbw
from abs_methods import weightings_registry as reg

PAIR = reg.MESH_BLOCK_PAIRS_BY_KEY["POA2026_LGA2011_C2021"]
RENTED = mbw.RENTED_ESTIMATED.count_type


# ------------------------------------------------------------- registry


def test_pair_is_named_poa_first_with_both_editions_and_the_census():
    assert PAIR.output_filename == "POA2026_LGA2011_C2021_Weightings.csv"
    assert PAIR.geog1_name == "Postcode (2026)"
    assert PAIR.geog2_name == "Local Government Area (2011)"


def test_mesh_block_pair_refuses_a_column_without_its_edition():
    with pytest.raises(ValueError, match="edition"):
        dataclasses.replace(PAIR, geog2_name="Local Government Area")


def test_mesh_block_pairs_are_kept_apart_from_tablebuilder_pairs():
    assert PAIR.key not in reg.PAIRS_BY_KEY


# --------------------------------------------------------------- counts


def test_read_mesh_block_counts_skips_titles_footers_and_contents(tmp_path):
    path = tmp_path / "counts.xlsx"
    titles = pd.DataFrame({"a": ["Australian Bureau of Statistics"] * 5})
    data = pd.DataFrame(
        {
            "MB_CODE_2021": ["10000010000", "10000009499", None, "(c) ABS"],
            "MB_CATEGORY_NAME_2021": ["Residential", "NOUSUALRESIDENCE", None, None],
            "AREA_ALBERS_SQKM": [0.1, None, None, None],
            "Dwelling": [44, None, None, None],
            "Person": [63, 11, None, None],
            "State": [1, 1, None, None],
        }
    )
    with pd.ExcelWriter(path) as writer:
        titles.to_excel(writer, sheet_name="Contents", index=False)
        for sheet in ("Table 1", "Table 1.1"):
            titles.to_excel(writer, sheet_name=sheet, index=False, header=False)
            data.to_excel(writer, sheet_name=sheet, index=False, startrow=6)
            data = data.assign(
                MB_CODE_2021=["10000020000", "10000029499", None, "(c) ABS"]
            )

    counts = mbw.read_mesh_block_counts(path)

    assert list(counts.columns) == ["MB_CODE_2021", "Dwellings", "Persons"]
    assert len(counts) == 4
    assert counts["MB_CODE_2021"].iloc[0] == "10000010000"
    assert counts["Dwellings"].sum() == 88
    assert counts["Persons"].sum() == 148


def test_rented_estimate_spreads_each_sa1_by_dwellings():
    counts = pd.DataFrame(
        {
            "MB_CODE_2021": ["m1", "m2", "m3", "m4"],
            "Dwellings": [30.0, 10.0, 0.0, 0.0],
            "Persons": [60.0, 20.0, 5.0, 0.0],
        }
    )
    mb_to_sa1 = pd.DataFrame(
        {"MB_CODE_2021": ["m1", "m2", "m3", "m4"], "SA1_CODE_2021": list("AABC")}
    )
    sa1_rented = pd.Series({"A": 8.0, "B": 3.0, "C": 0.0})

    result = mbw.estimate_rented_households(counts, mb_to_sa1, sa1_rented)

    assert result[RENTED].tolist() == [6.0, 2.0, 0.0, 0.0]
    # SA1 B has renters but no dwellings to put them in: they are dropped,
    # not turned into NaN.
    assert not result[RENTED].isna().any()
    assert "SA1_CODE_2021" not in result.columns


def test_carry_to_new_mesh_blocks_splits_by_ratio_and_keeps_totals():
    counts = pd.DataFrame(
        {"MB_CODE_2021": ["a", "b"], "Persons": [100.0, 10.0]}
    )
    correspondence = pd.DataFrame(
        {
            "MB_CODE_2021": ["a", "a", "b"],
            "MB_CODE_2026": ["x", "y", "y"],
            "RATIO_FROM_TO": [0.25, 0.75, 1.0],
        }
    )

    carried = mbw.carry_to_new_mesh_blocks(counts, correspondence, ["Persons"])

    assert carried.set_index("MB_CODE_2026")["Persons"].to_dict() == {
        "x": 25.0,
        "y": 85.0,
    }


def test_state_labels_add_the_state_only_where_a_name_repeats():
    regions = pd.DataFrame(
        {
            "LGA_NAME11": ["Campbelltown (C)", "Gosford (C)", "Campbelltown (C)"],
            "STE_CODE11": ["1", "1", "4"],
        }
    )

    labels = mbw.state_labels(regions, "LGA_NAME11", "STE_CODE11")

    assert labels.tolist() == [
        "Campbelltown (C) (NSW)",
        "Gosford (C)",
        "Campbelltown (C) (SA)",
    ]


# -------------------------------------------------------------- overlay


def km_box(x0, y0, x1, y1):
    """A box in kilometres, so areas clear the one-hectare uncovered floor."""
    return box(x0 * 1000, y0 * 1000, x1 * 1000, y1 * 1000)


UNINCORPORATED = {"1": "19399", "9": "99399"}


@pytest.fixture
def layout():
    """Two councils in state 1, an uncovered strip, and the sea below y=0.

    In kilometres. State 1 is [0, 30] x [0, 10]: council A is [0, 10],
    council B is [10, 20], and [20, 30] belongs to no council
    (Unincorporated, 19399). Below y = 0 is outside every state.
    """
    crs = "EPSG:9473"
    regions = gpd.GeoDataFrame(
        {"CODE": ["A", "B"], "STATE": ["1", "1"]},
        geometry=[km_box(0, 0, 10, 10), km_box(10, 0, 20, 10)],
        crs=crs,
    )
    states = gpd.GeoDataFrame(
        {"STATE": ["1"]}, geometry=[km_box(0, 0, 30, 10)], crs=crs
    )
    mesh_blocks = gpd.GeoDataFrame(
        {
            "MB_CODE": [
                "inside", "straddle", "uninc", "border", "coast", "sea",
                "island",
            ],
            "STATE": ["1"] * 6 + ["9"],
        },
        geometry=[
            km_box(1, 1, 3, 3),  # wholly in A
            km_box(3, 4, 13, 5),  # 7 in A, 3 in B
            km_box(22, 1, 25, 4),  # wholly unincorporated
            km_box(16, 1, 26, 2),  # 4 in B, 6 unincorporated
            km_box(4, -6, 6, 4),  # 8 in A, 12 at sea
            km_box(4, -4, 6, -2),  # 2 km offshore: nearest council
            km_box(4, -900, 6, -890),  # far out, state with no councils
        ],
        crs=crs,
    )
    return mesh_blocks, regions, states


def test_overlay_assigns_each_mesh_block_to_its_largest_overlap(layout):
    mesh_blocks, regions, states = layout

    result = mbw.assign_by_largest_overlap(
        mesh_blocks, regions, states, UNINCORPORATED
    ).set_index("MB_CODE")

    assert result["CODE"].to_dict() == {
        "inside": "A",
        "straddle": "A",
        "uninc": "19399",
        "border": "19399",
        # The sea is no candidate, so a coastal mesh block stays with its
        # council even when most of it lies outside the 2011 coastline.
        "coast": "A",
        "sea": "A",
        # Further than 5 km from any council in its state (Norfolk Island):
        # its state's Unincorporated region.
        "island": "99399",
    }
    assert result.loc["inside", "SHARE"] == pytest.approx(1.0)
    assert result.loc["straddle", "SHARE"] == pytest.approx(0.7)
    assert result.loc["border", "SHARE"] == pytest.approx(0.6)
    assert result.loc["coast", "SHARE"] == pytest.approx(0.4)
    assert result.loc["straddle", "CANDIDATES"] == 2
    assert result.loc["sea", "FALLBACK"]
    assert result.loc["island", "FALLBACK"]
    assert not result.loc["inside", "FALLBACK"]


def test_uncovered_parts_keep_real_gaps_and_drop_digitising_slivers(layout):
    _, regions, states = layout
    # A state edge drawn 20 cm beyond the councils' leaves a 4,000 m2
    # sliver along y = 10 km: below a hectare, so not Unincorporated land.
    states = states.set_geometry([box(0, 0, 30_000, 10_000.2)], crs=states.crs)

    parts = mbw.uncovered_parts(regions, states, UNINCORPORATED)

    assert len(parts) == 1
    assert parts["CODE"].tolist() == ["19399"]
    assert parts.geometry.area.iloc[0] == pytest.approx(100e6, rel=1e-3)


def test_overlay_drops_mesh_blocks_with_no_geometry(layout):
    mesh_blocks, regions, states = layout
    no_address = gpd.GeoDataFrame(
        {"MB_CODE": ["10000009499"], "STATE": ["1"]},
        geometry=[None],
        crs=mesh_blocks.crs,
    )
    mesh_blocks = pd.concat([mesh_blocks, no_address], ignore_index=True)

    result = mbw.assign_by_largest_overlap(
        mesh_blocks, regions, states, UNINCORPORATED
    )

    assert "10000009499" not in set(result["MB_CODE"])
    assert len(result) == 7


# ------------------------------------------------------------ weighting


def test_weightings_from_table_aggregates_weights_and_attaches_codes():
    table = pd.DataFrame(
        {
            "MB_CODE": ["1", "2", "3", "4", "5"],
            "CODE": ["10650", "10650", "18050", "18050", "18050"],
            "LGA_LABEL": ["Gosford (C)"] * 2 + ["Wyong (A)"] * 3,
            "POA_CODE_2026": ["2250", "2250", "2250", "2259", "ZZZZ"],
            "Persons": [60.0, 40.0, 50.0, 150.0, 7.0],
            "Dwellings": [20.0, 20.0, 20.0, 60.0, 3.0],
            RENTED: [5.0, 5.0, 4.0, 0.0, 1.0],
        }
    )

    df = mbw.weightings_from_table(table, PAIR)

    poa, lga = PAIR.geog1_name, PAIR.geog2_name
    assert set(df[poa]) == {"2250", "2259"}  # ZZZZ is not a postal area
    row = df.set_index([poa, lga]).loc[("2250", "Gosford (C)")]
    assert row["Persons"] == 100
    assert row[f"{poa} Fraction Persons"] == pytest.approx(100 / 150)
    assert row[f"{lga} Fraction Persons"] == pytest.approx(1.0)
    assert row["LGA_CODE11"] == "10650"
    wyong = df.set_index([poa, lga]).loc[("2250", "Wyong (A)")]
    assert wyong[f"{lga} Fraction Persons"] == pytest.approx(50 / 200)
    assert wyong["LGA_CODE11"] == "18050"
    # 2259 has no estimated renters: its fraction is NaN, not 0.
    other = df.set_index([poa, lga]).loc[("2259", "Wyong (A)")]
    assert np.isnan(other[f"{poa} Fraction {RENTED}"])
    assert other[f"{lga} Fraction {RENTED}"] == 0


def test_readme_explains_the_mesh_block_method_only_when_built():
    columns = [PAIR.geog1_name, PAIR.geog2_name, "Persons", "Dwellings", RENTED]
    with_mb = bw.render_readme({PAIR.key: pd.DataFrame(columns=columns)})
    without = bw.render_readme(
        {"POA2021_SAL2021_C2021": pd.DataFrame(columns=["Persons"])}
    )

    assert "## Files built from mesh blocks" in with_mb
    assert "Persons, Dwellings, Rented Households (Estimated)" in with_mb
    assert "mesh blocks" not in without
