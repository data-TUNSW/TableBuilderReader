"""TableBuilderReader on Australian Census and Temporary Entrants (ACTEID) extracts.

ACTEID tables differ from Census ones: the count is "Weighted count of Person
level" with decimal values, the geography is labelled "Postal Areas (EN)"
(2021) or "POAUCD Postal Areas (EN)" (2016) rather than "POA (EN)", every count
column is followed by "- RSE" and "- Annotations" columns, and a Symbol /
Description block sits in the footer. The reader keeps the RSE as a column and
drops the Annotations.
"""

import csv
from collections import Counter

import pytest

from abs_methods import table_builder_reader as tbr

POA_2021 = """Australian Bureau of Statistics

"Australian Census and Temporary Entrants, 2021"
"Postal Areas (EN) and VISAP Visa Subclass by TENLLD Tenure and landlord type"
"Counting: Weighted count of Person level"

Filters:
"Default Summation","Weighted count of Person level (#)"
"TENLLD Tenure and landlord type","Rented: Real estate agent"

,"Counting","Weighted count of Person level","Weighted count of Person level - RSE","Weighted count of Person level - Annotations",
"Postal Areas (EN)","VISAP Visa Subclass",
"2000, NSW","Temporary Student",5290.5,1.25,,
,"NZ Citizen (subclass 444)",120.4,0.00,,
"2010, NSW","Temporary Student",10.5,2.50,"^",
,"NZ Citizen (subclass 444)",0,0.00,,
"Migratory - Offshore - Shipping (Aust.)","Temporary Student",0,0.00,,
,"NZ Citizen (subclass 444)",0,0.00,,
"Total","Temporary Student",5301,0.00,,
,"NZ Citizen (subclass 444)",120.4,0.00,,

"Dataset: Australian Census and Temporary Entrants, 2021, TableBuilder"

"INFO","Cells in this table have been randomly adjusted to avoid the release of confidential data."
"INFO","Continuous variables in this table have been randomly adjusted to avoid the release of confidential data."


Symbol,Description
"#","No scaling applied"


"Copyright Commonwealth of Australia, 2026, see abs.gov.au/copyright"
"ABS data licensed under Creative Commons, see abs.gov.au/ccby"
"""

STE_2016 = """Australian Bureau of Statistics

"Australian Census and Temporary Entrants, 2016"
"SA1UCD Main Statistical Area Structure (Main ASGS) (EN), SEXP Sex and VISAP Visa Type by LFSP Labour force status (hierarchical) by TENLLD Tenure and landlord type"
"Counting: Weighted count of Person level"

Filters:
"Default summation","Weighted count of Person level (#)"
"TENLLD Tenure and landlord type","Rented: Real estate agent"

,,"LFSP Labour force status (hierarchical)","Employed, worked full-time","Employed, worked full-time - RSE","Employed, worked full-time - Annotations","Not in the labour force","Not in the labour force - RSE","Not in the labour force - Annotations","Overseas visitor","Overseas visitor - RSE","Overseas visitor - Annotations","Total","Total - RSE","Total - Annotations",
"SA1UCD Main Statistical Area Structure (Main ASGS) (EN)","SEXP Sex","VISAP Visa Type",
"New South Wales","Male","Student (500)",100.5,10.00,,20.25,20.00,,0,0.00,,120.75,9.00,,
,,"Not applicable",0,0.00,,0,0.00,,0,0.00,,0,0.00,,
,"Female","Student (500)",80,11.00,,30,21.00,,0,0.00,,110,10.00,,
,,"Not applicable",0,0.00,,0,0.00,,0,0.00,,0,0.00,,
"Total","Male","Student (500)",100.5,10.00,,20.25,20.00,,0,0.00,,120.75,9.00,,
,,"Not applicable",0,0.00,,0,0.00,,0,0.00,,0,0.00,,
,"Female","Student (500)",80,11.00,,30,21.00,,0,0.00,,110,10.00,,
,,"Not applicable",0,0.00,,0,0.00,,0,0.00,,0,0.00,,

"Data source: Australian Census and Temporary Entrants, 2016, TableBuilder"

"INFO","Cells in this table have been randomly adjusted to avoid the release of confidential data."


Symbol,Description
"#","No scaling applied"


"Copyright Commonwealth of Australia, 2026, see abs.gov.au/copyright"
"ABS data licensed under Creative Commons, see abs.gov.au/ccby"
"""


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_poa_extract_reads_weighted_counts_and_rse(tmp_path):
    path = write(tmp_path, "2021_TBP_POA_VISAP_TENLLD-REA.csv", POA_2021)
    reader = tbr.TableBuilderReader(path)
    df = reader.df
    assert reader.acteid
    assert list(df.columns) == ["POA", "VISAP", "Persons", "RSE"]
    assert sorted(df["POA"].unique()) == ["2000", "2010"]  # Total, Migratory dropped
    assert len(df) == 4
    assert df["Persons"].sum() == pytest.approx(5290.5 + 120.4 + 10.5)
    row = df[(df["POA"] == "2010") & (df["VISAP"] == "Temporary Student")]
    assert row["RSE"].item() == 2.5
    assert reader.filters == {"TENLLD": "REA"}


def test_ste_extract_detects_the_column_variable(tmp_path):
    path = write(tmp_path, "2016_TBP_STE_SEXP_VISAP_LFSP_TENLLD-REA.csv", STE_2016)
    reader = tbr.TableBuilderReader(path)
    df = reader.df
    assert reader.column_variable == "LFSP"
    assert list(df.columns) == ["STE", "SEXP", "VISAP", "LFSP", "Persons", "RSE"]
    # Total and Overseas visitor columns dropped, Total state block dropped.
    assert set(df["LFSP"]) == {"Employed, worked full-time", "Not in the labour force"}
    assert set(df["STE"]) == {"New South Wales"}
    assert df["Persons"].sum() == pytest.approx(100.5 + 20.25 + 80 + 30)
    nilf_m = df[(df["SEXP"] == "Male") & (df["LFSP"] == "Not in the labour force")
                & (df["VISAP"] == "Student (500)")]
    assert nilf_m["RSE"].item() == 20.0


def test_explicit_column_variable_is_honoured(tmp_path):
    path = write(tmp_path, "2016_TBP_STE_SEXP_VISAP_LFSP_TENLLD-REA.csv", STE_2016)
    df = tbr.TableBuilderReader(path, column_variable="LFSP").df
    assert list(df.columns) == ["STE", "SEXP", "VISAP", "LFSP", "Persons", "RSE"]


def test_wrong_column_variable_is_refused(tmp_path):
    path = write(tmp_path, "2016_TBP_STE_SEXP_VISAP_LFSP_TENLLD-REA.csv", STE_2016)
    with pytest.raises(ValueError, match="LFSP"):
        tbr.TableBuilderReader(path, column_variable="SEXP")


def test_census_extract_is_not_acteid(tmp_path):
    from test_reader_chunks import BODY, FOOTER, HEADER

    name = "2021_TBD_POA_SAL-OT_DWTD-Occupied.csv"
    path = write(tmp_path, name, HEADER + BODY + FOOTER)
    reader = tbr.TableBuilderReader(path)
    assert not reader.acteid
    assert "RSE" not in reader.df.columns


# ----------------------------------------------------------------- realdata
RAW = {
    2021: r"G:\Shared drives\Data\ABS\2021 Census\raw",
    2016: r"G:\Shared drives\Data\ABS\2016 Census\raw",
}


def dropped(label):
    """The reader's default drops: Total, Migratory, Overseas visitor (any case)."""
    return "Total" in label or "Migratory" in label or "overseas visitor" in label.lower()


def parse_tb(path):
    """Copy of Bond Loss nilf_tables.parse_tb (the reference), as a key -> count
    Counter, with the reader's default drops applied to the row labels."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        if prev and prev[0] == "" and len(prev) > 2 and cur and cur[0] != "":
            head, names = prev, [c for c in cur if c != ""]
            start = i + 1
            break
    k = len(names)
    cols = [(j, c) for j, c in enumerate(head) if j >= k and c
            and not c.endswith(" - RSE") and not c.endswith(" - Annotations")
            and c not in ("Total", "Counting") and not c.startswith("LFSP ")]
    out, last = Counter(), [""] * k
    for r in rows[start:]:
        if not r or all(c == "" for c in r):
            break
        last = [r[j] if r[j] != "" else last[j] for j in range(k)]
        if any(dropped(x) for x in last):
            continue
        for j, c in cols:
            if dropped(c):
                continue
            out[c] += float(r[j]) if r[j] != "" else 0.0
            out["__all__"] += float(r[j]) if r[j] != "" else 0.0
    return out


@pytest.mark.realdata
@pytest.mark.parametrize("year", [2021, 2016])
@pytest.mark.parametrize(
    "stem, columns",
    [
        ("TBP_POA_VISAP_TENLLD-REA", ["POA", "VISAP", "Persons", "RSE"]),
        ("TBP_STE_SEXP_VISAP_LFSP_TENLLD-REA",
         ["STE", "SEXP", "VISAP", "LFSP", "Persons", "RSE"]),
    ],
)
def test_real_acteid_totals_match_parse_tb(year, stem, columns):
    path = rf"{RAW[year]}\{year}_{stem}.csv"
    reader = tbr.TableBuilderReader(path)
    df = reader.df
    assert reader.acteid
    assert list(df.columns) == columns
    expected = parse_tb(path)
    assert df["Persons"].sum() == pytest.approx(expected["__all__"], abs=1e-6)
    if "LFSP" in df:
        got = df.groupby("LFSP")["Persons"].sum()
        for lfsp, n in got.items():
            assert n == pytest.approx(expected[lfsp], abs=1e-6)
    assert df["RSE"].notna().all()
