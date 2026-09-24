import numpy as np
import pandas as pd
import pytest

from abs_methods import geographic_weightings as gw

GEOG1 = "Postcode"
GEOG2 = "Local Government Area (2025)"
RENTED = gw.UNIVERSES_BY_KEY["rented_households"]


def clean_frame():
    """2000 splits 30/10 across two LGAs; 2007 sits wholly in Sydney."""
    return pd.DataFrame(
        {
            GEOG1: ["2000", "2000", "2007"],
            GEOG2: ["Sydney", "Inner West", "Sydney"],
            "Rented Households": [30, 10, 20],
            f"{GEOG1} Total Rented Households": [40, 40, 20],
            f"{GEOG2} Total Rented Households": [50, 10, 50],
            f"{GEOG1} Fraction Rented Households": [0.75, 0.25, 1.0],
            f"{GEOG2} Fraction Rented Households": [0.6, 1.0, 0.4],
        }
    )


def test_clean_frame_passes():
    assert gw.check_weightings(clean_frame(), GEOG1, GEOG2, [RENTED]) is None


def test_duplicate_pairs_raise():
    """Review Focus 3: two chunks covering the same pair double-count it."""
    df = pd.concat([clean_frame(), clean_frame().head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        gw.check_weightings(df, GEOG1, GEOG2, [RENTED])


def test_fractions_that_do_not_sum_to_one_raise():
    df = clean_frame()
    df.loc[0, f"{GEOG1} Fraction Rented Households"] = 0.9
    with pytest.raises(ValueError, match="sum to 1"):
        gw.check_weightings(df, GEOG1, GEOG2, [RENTED])


def test_negative_counts_raise():
    df = clean_frame()
    df.loc[0, "Rented Households"] = -1
    with pytest.raises(ValueError, match="negative"):
        gw.check_weightings(df, GEOG1, GEOG2, [RENTED])


def test_count_exceeding_its_total_raises():
    df = clean_frame()
    df.loc[0, "Rented Households"] = 999
    with pytest.raises(ValueError, match="exceeds"):
        gw.check_weightings(df, GEOG1, GEOG2, [RENTED])


def test_fraction_outside_zero_to_one_raises():
    df = clean_frame()
    df.loc[0, f"{GEOG2} Fraction Rented Households"] = 1.5
    with pytest.raises(ValueError, match="outside"):
        gw.check_weightings(df, GEOG1, GEOG2, [RENTED])


def test_geography_with_a_zero_total_is_exempt_from_the_sum_check():
    """A postcode with no renters sums to NaN, not 1, and that is correct."""
    df = clean_frame()
    extra = pd.DataFrame(
        {
            GEOG1: ["2008"],
            GEOG2: ["Sydney"],
            "Rented Households": [0],
            f"{GEOG1} Total Rented Households": [0],
            f"{GEOG2} Total Rented Households": [50],
            f"{GEOG1} Fraction Rented Households": [np.nan],
            f"{GEOG2} Fraction Rented Households": [0.0],
        }
    )
    df = pd.concat([df, extra], ignore_index=True)

    assert gw.check_weightings(df, GEOG1, GEOG2, [RENTED]) is None


def test_every_problem_is_reported_not_just_the_first():
    df = clean_frame()
    df.loc[0, "Rented Households"] = -1
    df.loc[1, f"{GEOG1} Fraction Rented Households"] = 1.5
    with pytest.raises(ValueError) as caught:
        gw.check_weightings(df, GEOG1, GEOG2, [RENTED])
    assert "negative" in str(caught.value)
    assert "outside" in str(caught.value)
