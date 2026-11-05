from pathlib import Path

import pandas as pd
import pytest

from abs_methods import build_weightings as bw
from abs_methods import weightings_registry as reg


@pytest.fixture
def fake_extracts(tmp_path, monkeypatch):
    """Two rented-household chunks for one postcode that crosses a border.

    Postcode 0800 appears in both chunks, against different suburbs. Weighted
    per chunk it would sum to 1 twice; weighted after concatenation it sums to
    1 once, which is the point of the rule.
    """
    chunks = {
        "chunk-a.csv": pd.DataFrame(
            {"POA": [800, 2000], "SAL": ["Darwin", "Sydney"], "Dwellings": [30, 40]}
        ),
        "chunk-b.csv": pd.DataFrame(
            {"POA": [800], "SAL": ["Larrakeyah"], "Dwellings": [10]}
        ),
    }
    for name, frame in chunks.items():
        frame.to_csv(tmp_path / name, index=False)

    class FakeReader:
        def __init__(self, file_name, add_filter=None, **kwargs):
            self.df = pd.read_csv(file_name)

    monkeypatch.setattr(bw.tbr, "TableBuilderReader", FakeReader)
    return tmp_path


def test_read_universe_concatenates_every_chunk(fake_extracts):
    """read_universe reads and concatenates all chunks into one frame.

    This does not by itself pin the concatenate-before-weighting rule - no
    totals are computed here. That invariant is pinned end to end by
    test_build_pair_weights_cross_border_postcode_correctly below, which
    calls build_pair and checks the resulting fractions.
    """
    df = bw.read_universe(
        sorted(fake_extracts.glob("chunk-*.csv")),
        count_type="Dwellings",
        geog1="POA",
        geog2="SAL",
    )
    assert len(df) == 3
    assert df[df["POA"] == "0800"]["Dwellings"].sum() == 40


def test_read_universe_zero_pads_postcodes(fake_extracts):
    df = bw.read_universe(
        sorted(fake_extracts.glob("chunk-*.csv")),
        count_type="Dwellings",
        geog1="POA",
        geog2="SAL",
    )
    assert set(df["POA"]) == {"0800", "2000"}


def test_build_pair_returns_none_when_no_extract_exists(tmp_path):
    pair = reg.PAIRS_BY_KEY["POA2021_SAL2021_C2021"]
    assert bw.build_pair(pair, raw_dir=tmp_path, out_dir=tmp_path, write=False) is None


def test_build_pair_writes_one_file_named_by_the_registry(
    fake_extracts, tmp_path, monkeypatch
):
    out_dir = tmp_path / "out"
    pair = reg.GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="SAL",
        geog2_edition=2021,
        geog2_name="Suburb (2021)",
        sources={"rented_households": "chunk-*.csv"},
    )

    df = bw.build_pair(pair, raw_dir=fake_extracts, out_dir=out_dir, write=True)

    assert (out_dir / "POA2021_SAL2021_C2021_Weightings.csv").exists()
    assert list(df.columns)[:2] == ["Postcode (2021)", "Suburb (2021)"]


def test_built_file_passes_its_own_qa(fake_extracts, tmp_path):
    pair = reg.GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="SAL",
        geog2_edition=2021,
        geog2_name="Suburb (2021)",
        sources={"rented_households": "chunk-*.csv"},
    )
    df = bw.build_pair(pair, raw_dir=fake_extracts, out_dir=tmp_path, write=False)

    by_postcode = df.groupby("Postcode (2021)")[
        "Postcode (2021) Fraction Rented Households"
    ].sum()
    assert by_postcode.tolist() == pytest.approx([1.0, 1.0])


def test_build_pair_weights_cross_border_postcode_correctly(fake_extracts, tmp_path):
    """Pins the concatenate-then-weight rule end to end via build_pair.

    Postcode 0800 has 30 dwellings in chunk-a (Darwin) and 10 in chunk-b
    (Larrakeyah). Weighted after concatenation, its national total is 40, so
    the two rows' Postcode Fraction values are 0.75 and 0.25 and sum to
    exactly 1.0. Weighted per chunk instead, each row would see only its own
    chunk's total (30 and 10), giving a fraction of 1.0 in each row and a sum
    of 2.0 - which is what test_chunks_are_concatenated_before_totals_are_
    computed (renamed above) could not have caught, since it never calls
    build_pair or looks at a fraction.
    """
    pair = reg.GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="SAL",
        geog2_edition=2021,
        geog2_name="Suburb (2021)",
        sources={"rented_households": "chunk-*.csv"},
    )
    df = bw.build_pair(pair, raw_dir=fake_extracts, out_dir=tmp_path, write=False)

    postcode_rows = df[df["Postcode (2021)"] == "0800"]
    fraction_column = "Postcode (2021) Fraction Rented Households"
    fractions = sorted(postcode_rows[fraction_column].tolist())
    assert fractions == pytest.approx([0.25, 0.75])
    assert (
        postcode_rows["Postcode (2021) Fraction Rented Households"].sum()
        == pytest.approx(1.0)
    )


def test_build_pair_warns_when_a_configured_shapefile_is_missing(
    fake_extracts, tmp_path, capsys
):
    """A configured-but-missing shapefile path prints a warning and the pair
    still builds, per the ruling that a None shapefile stays silent but a
    misconfigured one must not.
    """
    missing_shapefile = tmp_path / "does_not_exist.shp"
    pair = reg.GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="SAL",
        geog2_edition=2021,
        geog2_name="Suburb (2021)",
        sources={"rented_households": "chunk-*.csv"},
        geog1_shapefile=missing_shapefile,
        geog1_shapefile_name="POA_NAME21",
    )

    df = bw.build_pair(pair, raw_dir=fake_extracts, out_dir=tmp_path, write=False)

    captured = capsys.readouterr()
    assert "POA2021_SAL2021_C2021" in captured.out
    assert "Postcode (2021)" in captured.out
    assert str(missing_shapefile) in captured.out
    assert df is not None
    assert len(df) > 0


def test_build_pair_is_silent_when_no_shapefile_is_configured(
    fake_extracts, tmp_path, capsys
):
    """The complementary case: geog1_shapefile=None is a deliberate "this
    pair has none" and must not print anything about a missing shapefile.
    """
    pair = reg.GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="SAL",
        geog2_edition=2021,
        geog2_name="Suburb (2021)",
        sources={"rented_households": "chunk-*.csv"},
    )

    bw.build_pair(pair, raw_dir=fake_extracts, out_dir=tmp_path, write=False)

    captured = capsys.readouterr()
    assert "shapefile" not in captured.out.lower()


def test_readme_lists_each_built_file_and_its_universes():
    built = {
        "POA2021_SAL2021_C2021": pd.DataFrame(
            columns=["Postcode (2021)", "Suburb (2021)", "Rented Households"]
        )
    }
    readme = bw.render_readme(built)
    assert "POA2021_SAL2021_C2021_Weightings.csv" in readme
    assert "Rented Households" in readme
