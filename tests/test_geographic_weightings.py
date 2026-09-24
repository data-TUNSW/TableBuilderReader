import pandas as pd
import pytest

from abs_methods import geographic_weightings as gw


def test_universe_order_is_the_documented_block_order():
    assert [u.key for u in gw.UNIVERSES] == [
        "all_households",
        "all_persons",
        "rented_households",
        "renters_people",
        "private_rented_households",
        "private_renters_people",
    ]


def test_universes_by_key_covers_every_universe():
    assert set(gw.UNIVERSES_BY_KEY) == {u.key for u in gw.UNIVERSES}


def test_private_renter_universes_use_both_bases():
    assert gw.UNIVERSES_BY_KEY["private_rented_households"].count_type == "Dwellings"
    assert gw.UNIVERSES_BY_KEY["private_renters_people"].count_type == "Persons"


def test_universe_columns_are_suffixed_with_the_count_name():
    universe = gw.UNIVERSES_BY_KEY["rented_households"]
    assert gw.universe_columns(
        universe, "Postcode", "Local Government Area (2025)"
    ) == [
        "Rented Households",
        "Postcode Total Rented Households",
        "Local Government Area (2025) Total Rented Households",
        "Postcode Fraction Rented Households",
        "Local Government Area (2025) Fraction Rented Households",
    ]


@pytest.fixture
def two_postcode_frame():
    """Postcode 2000 splits across two LGAs; 2007 sits wholly in Sydney.

    Sydney LGA total is 30 + 20 = 50. Postcode 2000 total is 30 + 10 = 40.
    """
    return pd.DataFrame(
        {
            "POA": ["2000", "2000", "2007"],
            "LGA": ["Sydney", "Inner West", "Sydney"],
            "Dwellings": [30, 10, 20],
        }
    )


def test_geog1_fraction_is_the_share_of_geog1_falling_in_geog2(two_postcode_frame):
    out = gw.calculate_geog_weighting(
        two_postcode_frame,
        geog1="POA",
        geog1_name="Postcode",
        geog2="LGA",
        geog2_name="Local Government Area (2025)",
        count_type="Dwellings",
        count_name="Rented Households",
        add_shapefile_variables=False,
    )
    row = out[
        (out["Postcode"] == "2000")
        & (out["Local Government Area (2025)"] == "Sydney")
    ].iloc[0]
    assert row["Postcode Fraction"] == pytest.approx(30 / 40)


def test_geog2_fraction_is_the_share_of_geog2_made_up_by_geog1(two_postcode_frame):
    out = gw.calculate_geog_weighting(
        two_postcode_frame,
        geog1="POA",
        geog1_name="Postcode",
        geog2="LGA",
        geog2_name="Local Government Area (2025)",
        count_type="Dwellings",
        count_name="Rented Households",
        add_shapefile_variables=False,
    )
    row = out[
        (out["Postcode"] == "2000")
        & (out["Local Government Area (2025)"] == "Sydney")
    ].iloc[0]
    assert row["Local Government Area (2025) Fraction"] == pytest.approx(30 / 50)


def test_count_name_renames_the_count_and_total_columns(two_postcode_frame):
    out = gw.calculate_geog_weighting(
        two_postcode_frame,
        geog1="POA",
        geog1_name="Postcode",
        geog2="LGA",
        geog2_name="Local Government Area (2025)",
        count_type="Dwellings",
        count_name="Rented Households",
        add_shapefile_variables=False,
    )
    assert list(out.columns) == [
        "Postcode",
        "Local Government Area (2025)",
        "Rented Households",
        "Postcode Total Rented Households",
        "Local Government Area (2025) Total Rented Households",
        "Postcode Fraction",
        "Local Government Area (2025) Fraction",
    ]


def test_each_geography_fractions_sum_to_one(two_postcode_frame):
    out = gw.calculate_geog_weighting(
        two_postcode_frame,
        geog1="POA",
        geog1_name="Postcode",
        geog2="LGA",
        geog2_name="Local Government Area (2025)",
        count_type="Dwellings",
        count_name="Rented Households",
        add_shapefile_variables=False,
    )
    by_postcode = out.groupby("Postcode")["Postcode Fraction"].sum()
    assert by_postcode.tolist() == pytest.approx([1.0, 1.0])
