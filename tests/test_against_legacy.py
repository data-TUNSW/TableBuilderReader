import pandas as pd
import pytest

from abs_methods import weightings_registry as reg

LEGACY_DIR = reg.GEOGRAPHY_DIR / "processed"

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def combined_poa_lga():
    path = reg.WEIGHTINGS_DIR / "POA_LGA(2025)_Weightings.csv"
    if not path.exists():
        pytest.skip("run python -m abs_methods.build_weightings first")
    return pd.read_csv(path, dtype={"Postcode": str})


@pytest.fixture(scope="module")
def legacy_poa_lga_rented_households():
    return pd.read_csv(
        LEGACY_DIR / "POA_LGA(2025)_Rented_Household_Weightings.csv",
        dtype={"Postcode": str},
    )


def test_same_pairs_as_the_legacy_rented_household_file(
    combined_poa_lga, legacy_poa_lga_rented_households
):
    """Every legacy pair survives; the combined file may have more."""
    legacy_pairs = set(
        zip(
            legacy_poa_lga_rented_households["Postcode"],
            legacy_poa_lga_rented_households["Local Government Area (2025)"],
        )
    )
    combined_pairs = set(
        zip(
            combined_poa_lga["Postcode"],
            combined_poa_lga["Local Government Area (2025)"],
        )
    )
    assert legacy_pairs <= combined_pairs


def test_rented_household_counts_match_the_legacy_file(
    combined_poa_lga, legacy_poa_lga_rented_households
):
    keys = ["Postcode", "Local Government Area (2025)"]
    merged = legacy_poa_lga_rented_households[keys + ["Rented Households"]].merge(
        combined_poa_lga[keys + ["Rented Households"]],
        on=keys,
        suffixes=("_legacy", "_combined"),
    )
    assert len(merged) == len(legacy_poa_lga_rented_households)
    assert (merged["Rented Households_legacy"] == merged["Rented Households_combined"]).all()


def test_rented_household_fractions_match_the_legacy_file(
    combined_poa_lga, legacy_poa_lga_rented_households
):
    keys = ["Postcode", "Local Government Area (2025)"]
    merged = legacy_poa_lga_rented_households[keys + ["Postcode Fraction"]].merge(
        combined_poa_lga[keys + ["Postcode Fraction Rented Households"]],
        on=keys,
    )
    pd.testing.assert_series_equal(
        merged["Postcode Fraction"],
        merged["Postcode Fraction Rented Households"],
        check_names=False,
    )


def test_leading_zero_postcodes_survive_the_round_trip(combined_poa_lga):
    """0800 must still be 0800 after being written and read back."""
    assert combined_poa_lga["Postcode"].str.len().eq(4).all()
