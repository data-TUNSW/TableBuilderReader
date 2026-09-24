import pandas as pd
import pytest

from abs_methods import weightings_registry as reg


@pytest.fixture
def raw_dir(tmp_path):
    """Nine SAL chunks for one universe, plus a decoy from another."""
    for state in ("NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT", "OT"):
        (tmp_path / f"2021_TBD_POA_SAL-{state}_TEND-Rented.csv").touch()
    (tmp_path / "2021_TBD_POA_SAL-NSW_DWTD-Occupied.csv").touch()
    return tmp_path


def test_glob_resolves_every_chunk_of_one_universe(raw_dir):
    found = reg.resolve_sources("2021_TBD_POA_SAL-*_TEND-Rented.csv", raw_dir)
    assert len(found) == 9
    assert all("TEND-Rented" in p.name for p in found)


def test_glob_is_sorted_so_concatenation_is_deterministic(raw_dir):
    found = reg.resolve_sources("2021_TBD_POA_SAL-*_TEND-Rented.csv", raw_dir)
    assert found == sorted(found)


def test_an_unpulled_extract_resolves_to_nothing(raw_dir):
    """Review Focus 2: a universe with no extract yet is absent, not a crash."""
    assert reg.resolve_sources("2021_TBD_POA_SAL-*_TENLLD-*.csv", raw_dir) == []


def test_a_single_file_pattern_still_returns_a_list(raw_dir):
    found = reg.resolve_sources("2021_TBD_POA_SAL-NSW_DWTD-Occupied.csv", raw_dir)
    assert len(found) == 1


def test_postcodes_are_zero_padded_strings():
    """Review Focus 1: 0800 read as int 800 breaks every join."""
    series = pd.Series([800, 2000, 872])
    assert reg.normalise_geography_column(series, "POA").tolist() == [
        "0800",
        "2000",
        "0872",
    ]


def test_postcodes_already_strings_are_left_alone():
    series = pd.Series(["0800", "2000"])
    assert reg.normalise_geography_column(series, "POA").tolist() == [
        "0800",
        "2000",
    ]


def test_sa1_codes_are_padded_to_eleven():
    series = pd.Series([10102100701])
    assert reg.normalise_geography_column(series, "SA1").tolist() == [
        "10102100701"
    ]


def test_name_geographies_are_stripped_not_padded():
    series = pd.Series(["Sydney ", " Inner West"])
    assert reg.normalise_geography_column(series, "LGA").tolist() == [
        "Sydney",
        "Inner West",
    ]


# --- Sequence-of-patterns support (ruling: resolve_sources takes str | Sequence[str]) ---


def test_a_list_of_two_exact_filenames_resolves_to_both_sorted(raw_dir):
    names = [
        "2021_TBD_POA_SAL-VIC_TEND-Rented.csv",
        "2021_TBD_POA_SAL-NSW_TEND-Rented.csv",
    ]
    found = reg.resolve_sources(names, raw_dir)
    assert [p.name for p in found] == sorted(names)


def test_overlapping_patterns_in_a_list_deduplicate_to_one_file(raw_dir):
    patterns = [
        "2021_TBD_POA_SAL-NSW_TEND-Rented.csv",
        "2021_TBD_POA_SAL-NSW_*.csv",
    ]
    found = reg.resolve_sources(patterns, raw_dir)
    names = [p.name for p in found]
    assert names.count("2021_TBD_POA_SAL-NSW_TEND-Rented.csv") == 1


def test_a_list_with_a_nonmatching_pattern_still_resolves_the_others(raw_dir):
    patterns = [
        "2021_TBD_POA_SAL-NSW_TEND-Rented.csv",
        "does-not-exist-*.csv",
    ]
    found = reg.resolve_sources(patterns, raw_dir)
    assert [p.name for p in found] == ["2021_TBD_POA_SAL-NSW_TEND-Rented.csv"]


def test_a_bare_state_glob_is_too_greedy_which_is_why_lists_exist(tmp_path):
    """Path.glob lets * match across underscores, so a state-chunk glob
    intended for one universe also matches other universes' tenure-type
    extracts for that same state. Concatenating them silently inflates a
    universe's totals without tripping the fractions-sum-to-1 QA check.
    An explicit list of filenames avoids the trap; a wildcard does not.
    """
    for name in (
        "2021_TBP_POA_SAL-NSW.csv",
        "2021_TBP_POA_SAL-NSW_TEND-Rented.csv",
        "2021_TBP_POA_SAL-NSW_TENLLD-Rented Real estate agent.csv",
    ):
        (tmp_path / name).touch()

    explicit = reg.resolve_sources(["2021_TBP_POA_SAL-NSW.csv"], tmp_path)
    assert len(explicit) == 1

    greedy = reg.resolve_sources("2021_TBP_POA_SAL-*.csv", tmp_path)
    assert len(greedy) == 3
