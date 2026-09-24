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
