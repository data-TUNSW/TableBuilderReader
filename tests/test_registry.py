import dataclasses

import pytest

from abs_methods import geographic_weightings as gw
from abs_methods import weightings_registry as reg


def test_every_expected_pair_is_registered():
    assert {p.key for p in reg.PAIRS} == {
        "POA2021_LGA2025_C2021",
        "POA2021_SED2025_C2021",
        "POA2021_CED2025_C2021",
        "POA2021_SA32021_C2021",
        "POA2021_SAL2021_C2021",
        "SA12021_LGA2021_C2021",
    }


def test_output_filenames_carry_both_editions_and_the_census():
    assert (
        reg.PAIRS_BY_KEY["POA2021_LGA2025_C2021"].output_filename
        == "POA2021_LGA2025_C2021_Weightings.csv"
    )


def test_a_new_edition_gets_its_own_file():
    """The point of versioning: LGA 2026 must not overwrite LGA 2025."""
    current = reg.PAIRS_BY_KEY["POA2021_LGA2025_C2021"]
    later = dataclasses.replace(
        current,
        geog2_edition=2026,
        geog2_name="Local Government Area (2026)",
    )
    assert later.output_filename != current.output_filename


def test_every_geography_column_names_its_edition():
    for pair in reg.PAIRS:
        assert pair.geog1_name.endswith(f"({pair.geog1_edition})"), pair.key
        assert pair.geog2_name.endswith(f"({pair.geog2_edition})"), pair.key


def test_a_column_name_without_its_edition_is_refused():
    with pytest.raises(ValueError, match="edition"):
        reg.GeographyPair(
            geog1="POA",
            geog1_edition=2021,
            geog1_name="Postcode",
            geog2="LGA",
            geog2_edition=2025,
            geog2_name="Local Government Area (2025)",
            sources={},
        )


def test_a_column_name_with_the_wrong_edition_is_refused():
    with pytest.raises(ValueError, match="edition"):
        reg.GeographyPair(
            geog1="POA",
            geog1_edition=2026,
            geog1_name="Postcode (2021)",
            geog2="LGA",
            geog2_edition=2025,
            geog2_name="Local Government Area (2025)",
            sources={},
        )


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
    sources = reg.PAIRS_BY_KEY["POA2021_SAL2021_C2021"].sources
    for key, pattern in sources.items():
        if key == "all_persons":
            continue
        assert "SAL-*" in pattern


def test_poa_sal_covers_all_six_universes():
    sources = reg.PAIRS_BY_KEY["POA2021_SAL2021_C2021"].sources
    assert set(sources) == set(gw.UNIVERSES_BY_KEY)


def test_poa_lga_lists_every_universe():
    """Ruling 1: resolve_sources returning [] is already the mechanism for an
    unpulled extract, so POA2021_LGA2025_C2021 gets all six universe globs, exactly
    like every other pair - not a subset filtered down to the universes
    whose extracts happen to exist today.
    """
    assert set(reg.PAIRS_BY_KEY["POA2021_LGA2025_C2021"].sources) == set(
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
    source = reg.PAIRS_BY_KEY["POA2021_SAL2021_C2021"].sources["all_persons"]
    assert isinstance(source, list)
    assert len(source) == 9
    assert all("*" not in filename for filename in source)


def test_poa_sal_other_universes_remain_single_glob_strings():
    sources = reg.PAIRS_BY_KEY["POA2021_SAL2021_C2021"].sources
    for key, pattern in sources.items():
        if key == "all_persons":
            continue
        assert isinstance(pattern, str)
        assert "SAL-*" in pattern
