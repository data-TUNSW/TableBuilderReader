"""TableBuilderReader on per-state chunk files (``..._SAL-NSW_...``).

POA x SAL exceeds TableBuilder's 40m-cell cap, so it is pulled per state and
named ``2021_TB<D|P>_POA_SAL-<STATE><suffix>.csv``. The reader parses any
``-`` token in a filename as a ``<variable>-<category>`` filter, so it used to
read ``SAL-NSW`` as "SAL filtered to NSW", drop SAL from the variables and
return an empty frame. These tests pin the chunk suffix as part of the
geography token instead.
"""

import pytest

from abs_methods import build_weightings as bw
from abs_methods import table_builder_reader as tbr
from abs_methods import weightings_registry as reg

HEADER = """Australian Bureau of Statistics

"2021 Census - counting dwellings, place of enumeration"
"POA (EN) and SAL (EN) by DWTD Dwelling Type"
"Counting: Dwelling Records"

Filters:
"Default Summation","Dwelling Records"
"DWTD Dwelling Type","Occupied private dwellings"


"POA (EN)","SAL (EN)",
"""

BODY = """"2540, NSW","Jervis Bay",101,
,"Norfolk Island",0,
,"Migratory - Offshore - Shipping (OT)",0,
"2899, NSW","Jervis Bay",0,
,"Norfolk Island",1030,
,"Migratory - Offshore - Shipping (OT)",0,
"Total","Jervis Bay",101,
,"Norfolk Island",1030,
,"Migratory - Offshore - Shipping (OT)",0,
"""

FOOTER = """
"Dataset: Census of Population and Housing, 2021, TableBuilder"

"INFO","Cells in this table have been randomly adjusted."


"Copyright Commonwealth of Australia, 2026, see abs.gov.au/copyright"
"ABS data licensed under Creative Commons, see abs.gov.au/ccby"
"""


@pytest.fixture
def chunk_file(tmp_path):
    path = tmp_path / "2021_TBD_POA_SAL-OT_DWTD-Occupied.csv"
    path.write_text(HEADER + BODY + FOOTER, encoding="utf-8")
    return path


def test_chunk_suffix_is_not_a_filter(chunk_file):
    reader = tbr.TableBuilderReader(str(chunk_file))
    assert reader.filters == {"DWTD": "Occupied"}
    assert list(reader.variables) == ["POA", "SAL"]


def test_chunk_file_reads_its_rows(chunk_file):
    df = tbr.TableBuilderReader(str(chunk_file)).df
    assert list(df.columns) == ["POA", "SAL", "Dwellings"]
    # Total and Migratory rows are dropped by default, leaving 2 POAs x 2 SALs.
    assert len(df) == 4
    assert sorted(df["POA"].unique()) == ["2540", "2899"]
    assert df["Dwellings"].sum() == 1131


def test_chunk_states_match_registry():
    assert tbr.STATE_CHUNK_CODES == reg.CHUNK_STATES


def test_ordinary_filter_still_parsed(tmp_path):
    path = tmp_path / "2021_TBD_POA_SAL_DWTD-Occupied.csv"
    path.write_text(HEADER + BODY + FOOTER, encoding="utf-8")
    reader = tbr.TableBuilderReader(str(path))
    assert reader.filters == {"DWTD": "Occupied"}
    assert list(reader.variables) == ["POA", "SAL"]


# --- real per-state files on the shared drive --------------------------------

RAW = reg.RAW_DIR


def _real(name):
    path = RAW / name
    if not path.exists():
        pytest.skip(f"missing {path}")
    return path


@pytest.mark.realdata
@pytest.mark.parametrize(
    "name, count, n_sal",
    [
        ("2021_TBD_POA_SAL-OT_DWTD-Occupied.csv", "Dwellings", 5),
        ("2021_TBP_POA_SAL-ACT.csv", "Persons", 136),
    ],
)
def test_real_chunk_file(name, count, n_sal):
    reader = tbr.TableBuilderReader(str(_real(name)))
    df = reader.df
    assert list(df.columns) == ["POA", "SAL", count]
    # Every POA nationally is a row block in each state chunk; the reader drops
    # the Migratory - Offshore - Shipping POA by default, leaving 2,641.
    assert df["POA"].nunique() == 2641
    # The state's SALs, less its Migratory - Offshore - Shipping category.
    assert df["SAL"].nunique() == n_sal
    assert df[count].sum() > 0


@pytest.mark.realdata
def test_real_chunks_concatenate():
    paths = [
        _real("2021_TBD_POA_SAL-OT_DWTD-Occupied.csv"),
        _real("2021_TBD_POA_SAL-ACT_DWTD-Occupied.csv"),
    ]
    df = bw.read_universe(paths, "Dwellings", "POA", "SAL")
    assert {"POA", "SAL", "Dwellings"} <= set(df.columns)
    # Jervis Bay (OT) and Canberra suburbs (ACT) both survive the concat, and
    # postcode 2540 has dwellings in the OT chunk.
    assert (df["SAL"] == "Jervis Bay").any()
    assert (df["SAL"] == "Belconnen").any()
    assert df.loc[df["POA"] == "2540", "Dwellings"].sum() > 0
