import numpy as np
import pandas as pd
import pytest

from abs_methods import geographic_weightings as gw
from abs_methods import weightings_registry as reg

LEGACY_DIR = reg.GEOGRAPHY_DIR / "processed"

pytestmark = pytest.mark.realdata

#: POA_SAL has no extracts pulled yet, so build_all() correctly skips it and
#: writes no file for it. Every other registered pair is expected to build.
BUILT_PAIRS = [pair for pair in reg.PAIRS if pair.key != "POA_SAL"]

POA_PAIRS = [pair for pair in BUILT_PAIRS if pair.geog1 == "POA"]


def _combined_path(pair: reg.GeographyPair):
    return reg.WEIGHTINGS_DIR / pair.output_filename


def _skip_if_missing(path):
    if not path.exists():
        pytest.skip(
            f"missing {path.name}; run python -m abs_methods.build_weightings first"
        )


def _read_header(path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _fraction_sum_cases():
    """(pair, universe, side) for every universe present in every built file.

    Built once at collection time from each file's header only (cheap), so
    the resulting test ids name exactly which file/universe/geography-side
    combination is under test. If the weightings haven't been built yet this
    simply produces no cases; test_expected_weightings_files_exist is what
    reports that as a failure.
    """
    cases = []
    for pair in BUILT_PAIRS:
        path = _combined_path(pair)
        if not path.exists():
            continue
        header = _read_header(path)
        for universe in gw.UNIVERSES:
            count_column = gw.universe_columns(
                universe, pair.geog1_name, pair.geog2_name
            )[0]
            if count_column not in header:
                continue
            for side in ("geog1", "geog2"):
                cases.append((pair, universe, side))
    return cases


FRACTION_SUM_CASES = _fraction_sum_cases()
FRACTION_SUM_IDS = [
    f"{pair.key}-{universe.key}-{side}" for pair, universe, side in FRACTION_SUM_CASES
]


@pytest.fixture(scope="module")
def combined_frame_cache():
    """Lazily-read, cached full frames for the built pairs, keyed by pair.

    Geography code columns (Postcode, SA1) are read as strings so a leading
    zero postcode or SA1 code is not corrupted before it's even checked.
    """
    cache: dict[str, pd.DataFrame] = {}

    def _get(pair: reg.GeographyPair) -> pd.DataFrame:
        if pair.key not in cache:
            path = _combined_path(pair)
            dtype = {}
            if pair.geog1 in reg.GEOGRAPHY_CODE_WIDTHS:
                dtype[pair.geog1_name] = str
            if pair.geog2 in reg.GEOGRAPHY_CODE_WIDTHS:
                dtype[pair.geog2_name] = str
            cache[pair.key] = pd.read_csv(path, dtype=dtype or None)
        return cache[pair.key]

    return _get


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


# ---------------------------------------------------------------------------
# 1. The expected file set exists.
# ---------------------------------------------------------------------------


def test_expected_weightings_files_exist():
    """weightings/ has exactly one file per built pair plus the README.

    POA_SAL is deliberately excluded: its extracts are not pulled yet, so
    build_all() skips it and writes no POA_SAL(...)_Weightings.csv. If this
    ever changes to include a partial write, or a pair silently fails to
    write while the others still pass, this is the test that would catch it.
    """
    if not reg.WEIGHTINGS_DIR.exists():
        pytest.skip("run python -m abs_methods.build_weightings first")
    expected = {pair.output_filename for pair in BUILT_PAIRS}
    expected.add("README.md")
    actual = {path.name for path in reg.WEIGHTINGS_DIR.iterdir()}
    assert actual == expected


# ---------------------------------------------------------------------------
# 2. A second universe against legacy: POA_SED All Households.
#
# POA_LGA's tests above only ever compare the "Rented Households" universe.
# This compares "Households" on a different pair, which is what would catch
# a fraction column accidentally wired up to the wrong universe.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def combined_poa_sed():
    path = reg.WEIGHTINGS_DIR / "POA_SED(2025)_Weightings.csv"
    if not path.exists():
        pytest.skip("run python -m abs_methods.build_weightings first")
    return pd.read_csv(path, dtype={"Postcode": str})


@pytest.fixture(scope="module")
def legacy_poa_sed_all_households():
    return pd.read_csv(
        LEGACY_DIR / "POA_SED(2025)_All_Household_Weightings.csv",
        dtype={"Postcode": str},
    )


def test_same_pairs_as_the_legacy_all_household_file(
    combined_poa_sed, legacy_poa_sed_all_households
):
    legacy_pairs = set(
        zip(
            legacy_poa_sed_all_households["Postcode"],
            legacy_poa_sed_all_households["State Electorate (2025)"],
        )
    )
    combined_pairs = set(
        zip(
            combined_poa_sed["Postcode"],
            combined_poa_sed["State Electorate (2025)"],
        )
    )
    assert legacy_pairs <= combined_pairs


def test_household_counts_match_the_legacy_sed_file(
    combined_poa_sed, legacy_poa_sed_all_households
):
    keys = ["Postcode", "State Electorate (2025)"]
    merged = legacy_poa_sed_all_households[keys + ["Households"]].merge(
        combined_poa_sed[keys + ["Households"]],
        on=keys,
        suffixes=("_legacy", "_combined"),
    )
    assert len(merged) == len(legacy_poa_sed_all_households)
    assert (merged["Households_legacy"] == merged["Households_combined"]).all()


def test_household_fractions_match_the_legacy_sed_file(
    combined_poa_sed, legacy_poa_sed_all_households
):
    keys = ["Postcode", "State Electorate (2025)"]
    merged = legacy_poa_sed_all_households[keys + ["Postcode Fraction"]].merge(
        combined_poa_sed[keys + ["Postcode Fraction Households"]],
        on=keys,
    )
    pd.testing.assert_series_equal(
        merged["Postcode Fraction"],
        merged["Postcode Fraction Households"],
        check_names=False,
    )


# ---------------------------------------------------------------------------
# 3. Universe block order, on every built file.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", BUILT_PAIRS, ids=[p.key for p in BUILT_PAIRS])
def test_universe_block_order(pair):
    """Universe blocks appear in gw.UNIVERSES order, not just any order."""
    path = _combined_path(pair)
    _skip_if_missing(path)
    header = _read_header(path)

    present = []
    for universe in gw.UNIVERSES:
        count_column = gw.universe_columns(
            universe, pair.geog1_name, pair.geog2_name
        )[0]
        if count_column in header:
            present.append((universe, header.index(count_column)))

    for (earlier, earlier_pos), (later, later_pos) in zip(present, present[1:]):
        assert earlier_pos < later_pos, (
            f"{pair.output_filename}: universe block order violated - "
            f"{earlier.count_name!r} (column {earlier_pos}) should appear "
            f"before {later.count_name!r} (column {later_pos}) per "
            "gw.UNIVERSES, but does not"
        )


# ---------------------------------------------------------------------------
# 4 & 5. The fraction-sum invariant and the NaN convention, on every built
# file, every universe present in it, and both geography sides.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair,universe,side", FRACTION_SUM_CASES, ids=FRACTION_SUM_IDS)
def test_fractions_sum_to_one_within_each_live_geography(
    pair, universe, side, combined_frame_cache
):
    df = combined_frame_cache(pair)
    geog_name = pair.geog1_name if side == "geog1" else pair.geog2_name
    count = universe.count_name
    total_column = f"{geog_name} Total {count}"
    fraction_column = f"{geog_name} Fraction {count}"

    live = df.groupby(geog_name)[total_column].max() > 0
    sums = df.groupby(geog_name)[fraction_column].sum(min_count=1)
    live_sums = sums[live.reindex(sums.index, fill_value=False)]
    off = live_sums[~np.isclose(live_sums, 1.0, atol=1e-9)]

    assert len(off) == 0, (
        f"{pair.output_filename}: universe {count!r}, {geog_name} side - "
        f"{len(off)} geographies' fractions do not sum to 1 "
        f"(worst offender: {off.abs().sub(1).idxmax() if len(off) else None})"
    )


@pytest.mark.parametrize("pair,universe,side", FRACTION_SUM_CASES, ids=FRACTION_SUM_IDS)
def test_fraction_nan_matches_zero_total(pair, universe, side, combined_frame_cache):
    df = combined_frame_cache(pair)
    geog_name = pair.geog1_name if side == "geog1" else pair.geog2_name
    count = universe.count_name
    total_column = f"{geog_name} Total {count}"
    fraction_column = f"{geog_name} Fraction {count}"

    fraction = df[fraction_column]
    total = df[total_column]

    nan_in_live = fraction.isna() & (total > 0)
    assert int(nan_in_live.sum()) == 0, (
        f"{pair.output_filename}: universe {count!r}, {geog_name} side - "
        f"{int(nan_in_live.sum())} rows have a NaN fraction where "
        f"{total_column} is above zero"
    )

    non_nan_in_dead = fraction.notna() & (total == 0)
    assert int(non_nan_in_dead.sum()) == 0, (
        f"{pair.output_filename}: universe {count!r}, {geog_name} side - "
        f"{int(non_nan_in_dead.sum())} rows have a non-NaN fraction where "
        f"{total_column} is zero"
    )


# ---------------------------------------------------------------------------
# 6. Postcode string integrity across all POA files.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", POA_PAIRS, ids=[p.key for p in POA_PAIRS])
def test_postcode_survives_round_trip_in_every_poa_file(pair):
    path = _combined_path(pair)
    _skip_if_missing(path)
    df = pd.read_csv(path, dtype={"Postcode": str})
    lengths = df["Postcode"].str.len()
    bad = lengths[lengths != 4]
    assert bad.empty, (
        f"{pair.output_filename}: {len(bad)} Postcode values are not 4 "
        f"characters after the write/read round trip "
        f"(e.g. {df.loc[bad.index[0], 'Postcode']!r})"
    )
