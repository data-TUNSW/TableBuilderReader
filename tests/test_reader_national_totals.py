"""National totals from the column-variable path match a direct parse.

The Bond Loss cluster work found the reader's national total of primary-family
reference persons in ``2021_TBP_POA_RPIP_HCFMD(2 digit)_BPLP(2 digit)_TENLLD-REA.csv``
was about half a direct parse of the same file (1.18m against 2.39m), while
totals over a subset of postcodes matched. The reader was right. TableBuilder
writes a national ``"Total"`` row block after the last postcode, and the direct
parse kept it, so it counted every person twice: 1,181,725 in the postcode
rows plus 1,210,777 in the Total row is 2,392,502. A postcode subset never
contains the Total row, which is why subsets agreed.

These tests pin that the reader drops the Total row block (and the Total and
Overseas visitor columns) on the column-variable path, so its national total is
the sum of the postcode cells.
"""

import csv
from collections import Counter

import pytest

from abs_methods import table_builder_reader as tbr

HEADER = """Australian Bureau of Statistics

"2021 Census - counting persons, place of enumeration"
"POA (EN), RPIP Family/Household Reference Person Indicator and HCFMD Family Household Composition (Dwelling) by BPLP Country of Birth of Person by TENLLD Tenure and Landlord Type"
"Counting: Person Records"

Filters:
"Default Summation","Person Records"
"TENLLD Tenure and Landlord Type","Rented: Real estate agent"

,,"BPLP Country of Birth of Person","Australia","India","Overseas visitor","Total",
"POA (EN)","RPIP Family/Household Reference Person Indicator","HCFMD Family Household Composition (Dwelling)",
"""

# Two postcodes, two RPIP categories, two HCFMD categories, then the national
# Total block that TableBuilder appends (its cells are adjusted, so they differ
# slightly from the sum of the postcode cells, as in the real file).
BODY = """"2000, NSW","Reference person in primary family","One family household: One parent family",10,5,0,15,
,,"Lone person household",0,0,0,0,
,"Other household member","One family household: One parent family",20,8,0,28,
,,"Lone person household",0,0,0,0,
"2010, NSW","Reference person in primary family","One family household: One parent family",7,3,0,10,
,,"Lone person household",0,0,0,0,
,"Other household member","One family household: One parent family",12,4,0,16,
,,"Lone person household",0,0,0,0,
"Migratory - Offshore - Shipping (NSW)","Reference person in primary family","One family household: One parent family",0,0,0,0,
,,"Lone person household",0,0,0,0,
,"Other household member","One family household: One parent family",0,0,0,0,
,,"Lone person household",0,0,0,0,
"Total","Reference person in primary family","One family household: One parent family",18,8,0,26,
,,"Lone person household",0,0,0,0,
,"Other household member","One family household: One parent family",33,12,0,45,
,,"Lone person household",0,0,0,0,
"""

FOOTER = """
"Dataset: Census of Population and Housing, 2021, TableBuilder"

"INFO","Cells in this table have been randomly adjusted."


"Copyright Commonwealth of Australia, 2026, see abs.gov.au/copyright"
"ABS data licensed under Creative Commons, see abs.gov.au/ccby"
"""

PRIMARY = "Reference person in primary family"
NAME = "2021_TBP_POA_RPIP_HCFMD_BPLP_TENLLD-REA.csv"
REAL = (
    r"G:\Shared drives\Data\ABS\2021 Census\raw"
    r"\2021_TBP_POA_RPIP_HCFMD(2 digit)_BPLP(2 digit)_TENLLD-REA.csv"
)


def direct_parse(path, keep_total_row=False):
    """Independent parse: {(postcode, RPIP): count} over the BPLP cells.

    Drops the Total and Overseas visitor columns, Migratory rows and (unless
    asked) the national Total row block.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    for i, row in enumerate(rows):
        if row and row[0] == "" and len(row) > 3 and row[2]:
            head, start = row, i + 2
            break
    out = Counter()
    poa = rpip = None
    for row in rows[start:]:
        if not row or not any(row):
            break
        poa = row[0] or poa
        rpip = row[1] or rpip
        if poa.startswith("Migratory") or (poa == "Total" and not keep_total_row):
            continue
        for col, val in zip(head[3:], row[3:]):
            if col in ("", "Total", "Overseas visitor"):
                continue
            out[(poa[:4] if poa != "Total" else "Total", rpip)] += int(val)
    return out


@pytest.fixture
def rpip_file(tmp_path):
    path = tmp_path / NAME
    path.write_text(HEADER + BODY + FOOTER, encoding="utf-8")
    return path


def test_total_row_block_is_dropped(rpip_file):
    df = tbr.TableBuilderReader(str(rpip_file), column_variable="BPLP").df
    assert sorted(df["POA"].unique()) == ["2000", "2010"]
    assert not df["BPLP"].isin(["Total", "Overseas visitor"]).any()


def test_national_total_is_sum_of_postcode_cells(rpip_file):
    df = tbr.TableBuilderReader(str(rpip_file), column_variable="BPLP").df
    reader = df.loc[df["RPIP"] == PRIMARY, "Persons"].sum()
    assert reader == 25  # 2000: 10 + 5, 2010: 7 + 3


def test_keeping_the_total_row_doubles_the_direct_parse(rpip_file):
    """The mechanism of the reported mismatch: not a reader bug."""
    with_total = direct_parse(rpip_file, keep_total_row=True)
    without = direct_parse(rpip_file)
    primary = lambda c: sum(v for (_, r), v in c.items() if r == PRIMARY)  # noqa: E731
    assert primary(without) == 25
    assert primary(with_total) == 25 + 26


@pytest.mark.realdata
def test_real_rpip_table_matches_direct_parse():
    """12.6m cells: about a minute and a few GB of memory."""
    expected = direct_parse(REAL)
    df = tbr.TableBuilderReader(REAL, column_variable="BPLP").df
    got = df.groupby(["POA", "RPIP"])["Persons"].sum()
    got = {k: v for k, v in got.items() if v}
    assert got == {k: v for k, v in expected.items() if v and k[1] != "Overseas visitor"}
    national = sum(v for (_, r), v in expected.items() if r == PRIMARY)
    assert national == 1_181_725
    assert direct_parse(REAL, keep_total_row=True)[("Total", PRIMARY)] == 1_210_777
