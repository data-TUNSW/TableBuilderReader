import numpy as np
import pandas as pd
import pytest

from abs_methods import geographic_weightings as gw

GEOG1 = "Postcode"
GEOG2 = "Local Government Area (2025)"


def weighting(pairs, count_name):
    """Build a single-universe frame the way calculate_geog_weighting does."""
    df = pd.DataFrame(pairs, columns=["POA", "LGA", "Dwellings"])
    return gw.calculate_geog_weighting(
        df,
        geog1="POA",
        geog1_name=GEOG1,
        geog2="LGA",
        geog2_name=GEOG2,
        count_type="Dwellings",
        count_name=count_name,
        add_shapefile_variables=False,
    )


@pytest.fixture
def mismatched_universes():
    """2008 has occupied dwellings but no renters at all.

    All Households covers 2000/Sydney, 2000/Inner West and 2008/Sydney.
    Rented Households covers only the two 2000 pairs.
    """
    return {
        "all_households": weighting(
            [
                ("2000", "Sydney", 30),
                ("2000", "Inner West", 10),
                ("2008", "Sydney", 20),
            ],
            "Households",
        ),
        "rented_households": weighting(
            [("2000", "Sydney", 6), ("2000", "Inner West", 4)],
            "Rented Households",
        ),
    }


def test_blocks_appear_in_universe_order(mismatched_universes):
    combined = gw.combine_universes(mismatched_universes, GEOG1, GEOG2)
    assert list(combined.columns) == [
        GEOG1,
        GEOG2,
        "Households",
        f"{GEOG1} Total Households",
        f"{GEOG2} Total Households",
        f"{GEOG1} Fraction Households",
        f"{GEOG2} Fraction Households",
        "Rented Households",
        f"{GEOG1} Total Rented Households",
        f"{GEOG2} Total Rented Households",
        f"{GEOG1} Fraction Rented Households",
        f"{GEOG2} Fraction Rented Households",
    ]


def test_absent_universes_contribute_no_columns():
    combined = gw.combine_universes(
        {"rented_households": weighting([("2000", "Sydney", 6)], "Rented Households")},
        GEOG1,
        GEOG2,
    )
    assert not [c for c in combined.columns if "Households" in c and "Rented" not in c]


def test_outer_join_keeps_pairs_present_in_only_one_universe(mismatched_universes):
    combined = gw.combine_universes(mismatched_universes, GEOG1, GEOG2)
    pairs = set(zip(combined[GEOG1], combined[GEOG2]))
    assert ("2008", "Sydney") in pairs
    assert len(combined) == 3


def test_missing_count_fills_to_zero(mismatched_universes):
    combined = gw.combine_universes(mismatched_universes, GEOG1, GEOG2)
    row = combined[combined[GEOG1] == "2008"].iloc[0]
    assert row["Rented Households"] == 0


def test_geography_absent_from_a_universe_gets_zero_total_and_nan_fraction(
    mismatched_universes,
):
    """Review Focus 4: 0/0 is NaN, not 0, not inf, and not a warning."""
    combined = gw.combine_universes(mismatched_universes, GEOG1, GEOG2)
    row = combined[combined[GEOG1] == "2008"].iloc[0]

    assert row[f"{GEOG1} Total Rented Households"] == 0
    assert np.isnan(row[f"{GEOG1} Fraction Rented Households"])


def test_pair_absent_from_a_universe_whose_geography_is_present_gets_zero_fraction():
    """A real zero: the LGA has renters, this postcode contributes none."""
    frames = {
        "all_households": weighting(
            [("2000", "Sydney", 30), ("2008", "Sydney", 20)], "Households"
        ),
        "rented_households": weighting([("2000", "Sydney", 6)], "Rented Households"),
    }
    combined = gw.combine_universes(frames, GEOG1, GEOG2)
    row = combined[combined[GEOG1] == "2008"].iloc[0]

    assert row[f"{GEOG2} Total Rented Households"] == 6
    assert row[f"{GEOG2} Fraction Rented Households"] == 0.0


def test_present_fractions_are_untouched(mismatched_universes):
    combined = gw.combine_universes(mismatched_universes, GEOG1, GEOG2)
    row = combined[
        (combined[GEOG1] == "2000") & (combined[GEOG2] == "Sydney")
    ].iloc[0]
    assert row[f"{GEOG1} Fraction Households"] == pytest.approx(30 / 40)
    assert row[f"{GEOG1} Fraction Rented Households"] == pytest.approx(6 / 10)
