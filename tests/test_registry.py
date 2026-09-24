import pytest

from abs_methods import geographic_weightings as gw
from abs_methods import weightings_registry as reg


def test_every_expected_pair_is_registered():
    assert {p.key for p in reg.PAIRS} == {
        "POA_LGA(2025)",
        "POA_SED(2025)",
        "POA_CED(2025)",
        "POA_SA3",
        "POA_SAL",
        "SA1_LGA",
    }


def test_output_filenames_follow_the_convention():
    assert (
        reg.PAIRS_BY_KEY["POA_LGA(2025)"].output_filename
        == "POA_LGA(2025)_Weightings.csv"
    )
    assert reg.PAIRS_BY_KEY["POA_SA3"].output_filename == "POA_SA3_Weightings.csv"


def test_every_source_key_is_a_real_universe():
    for pair in reg.PAIRS:
        unknown = set(pair.sources) - set(gw.UNIVERSES_BY_KEY)
        assert not unknown, f"{pair.key} references unknown universes: {unknown}"


def test_poa_sal_sources_are_state_globs_not_literal_paths():
    """Nine chunks per universe; adding a state must not need a code change.

    all_persons is excluded here: ruling 3 makes it an explicit list of
    filenames rather than a glob, because ``SAL-*`` is ambiguous with the
    other chunked person-basis universes. See
    test_poa_sal_all_persons_is_explicit_state_list below.
    """
    sources = reg.PAIRS_BY_KEY["POA_SAL"].sources
    for key, pattern in sources.items():
        if key == "all_persons":
            continue
        assert "SAL-*" in pattern


def test_poa_sal_covers_all_six_universes():
    assert set(reg.PAIRS_BY_KEY["POA_SAL"].sources) == set(gw.UNIVERSES_BY_KEY)


def test_poa_lga_lists_every_universe():
    """Ruling 1: resolve_sources returning [] is already the mechanism for an
    unpulled extract, so POA_LGA(2025) gets all six universe globs, exactly
    like every other pair - not a subset filtered down to the universes
    whose extracts happen to exist today.
    """
    assert set(reg.PAIRS_BY_KEY["POA_LGA(2025)"].sources) == set(
        gw.UNIVERSES_BY_KEY
    )


def test_pairs_by_key_covers_every_pair():
    assert set(reg.PAIRS_BY_KEY) == {p.key for p in reg.PAIRS}


def test_poa_sal_all_persons_is_explicit_state_list():
    """Ruling 3: a ``SAL-*`` glob for all_persons would also match the
    per-tenure chunks of rented_households/renters_people/etc (their
    filenames continue past the state code with an underscore, e.g.
    ``2021_TBP_POA_SAL-NSW_TEND-Rented.csv``), silently sweeping other
    universes' records into all_persons' total. So all_persons is an
    explicit list of the nine per-state filenames instead of a glob.
    """
    source = reg.PAIRS_BY_KEY["POA_SAL"].sources["all_persons"]
    assert isinstance(source, list)
    assert len(source) == 9
    assert all("*" not in filename for filename in source)


def test_poa_sal_other_universes_remain_single_glob_strings():
    sources = reg.PAIRS_BY_KEY["POA_SAL"].sources
    for key, pattern in sources.items():
        if key == "all_persons":
            continue
        assert isinstance(pattern, str)
        assert "SAL-*" in pattern
