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


def test_chunks_are_concatenated_before_totals_are_computed(fake_extracts):
    """The spec's chunking rule: per-chunk totals would give 0800 two 1.0s."""
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
    pair = reg.PAIRS_BY_KEY["POA_SAL"]
    assert bw.build_pair(pair, raw_dir=tmp_path, out_dir=tmp_path, write=False) is None


def test_build_pair_writes_one_file_named_by_the_registry(
    fake_extracts, tmp_path, monkeypatch
):
    out_dir = tmp_path / "out"
    pair = reg.GeographyPair(
        key="POA_SAL",
        geog1="POA",
        geog1_name="Postcode",
        geog2="SAL",
        geog2_name="Suburb",
        sources={"rented_households": "chunk-*.csv"},
    )

    df = bw.build_pair(pair, raw_dir=fake_extracts, out_dir=out_dir, write=True)

    assert (out_dir / "POA_SAL_Weightings.csv").exists()
    assert list(df.columns)[:2] == ["Postcode", "Suburb"]


def test_built_file_passes_its_own_qa(fake_extracts, tmp_path):
    pair = reg.GeographyPair(
        key="POA_SAL",
        geog1="POA",
        geog1_name="Postcode",
        geog2="SAL",
        geog2_name="Suburb",
        sources={"rented_households": "chunk-*.csv"},
    )
    df = bw.build_pair(pair, raw_dir=fake_extracts, out_dir=tmp_path, write=False)

    by_postcode = df.groupby("Postcode")["Postcode Fraction Rented Households"].sum()
    assert by_postcode.tolist() == pytest.approx([1.0, 1.0])


def test_readme_lists_each_built_file_and_its_universes():
    built = {
        "POA_SAL": pd.DataFrame(
            columns=["Postcode", "Suburb", "Rented Households"]
        )
    }
    readme = bw.render_readme(built)
    assert "POA_SAL_Weightings.csv" in readme
    assert "Rented Households" in readme
