"""Stacked-column labels and count codes on the column-variable path.

With ``column_variable`` set, the reader stacks the column variable into the
last position of the long frame. It used to label the columns in file-name
order, so when the column variable was not last in the name (``..._LFSP_SEXP_...``
with LFSP across the columns) the LFSP label landed on the SEXP column. Bond
Loss worked around this with a ``ColumnLastReader`` subclass, which also gave
the families dataset (``TBF``) a count name.
"""

import pytest

from abs_methods import table_builder_reader as tbr

RAW = r"G:\Shared drives\Data\ABS\2021 Census\raw"

FOOTER = """
"Dataset: Census of Population and Housing, 2021, TableBuilder"

"INFO","Cells in this table have been randomly adjusted."


"Copyright Commonwealth of Australia, 2026, see abs.gov.au/copyright"
"ABS data licensed under Creative Commons, see abs.gov.au/ccby"
"""

PERSONS = """Australian Bureau of Statistics

"2021 Census - counting persons, place of enumeration"
"POA (EN) and SEXP Sex by LFSP Labour Force Status by TENLLD Tenure and Landlord Type"
"Counting: Person Records"

Filters:
"Default Summation","Person Records"
"TENLLD Tenure and Landlord Type","Rented: Real estate agent"

,"LFSP Labour Force Status","Employed, worked full-time","Not in the labour force","Total",
"POA (EN)","SEXP Sex",
"2000, NSW","Male",10,2,12,
,"Female",8,5,13,
"2010, NSW","Male",4,1,5,
,"Female",3,6,9,
"Total","Male",14,3,17,
,"Female",11,11,22,
"""

FAMILIES = """Australian Bureau of Statistics

"2021 Census - counting families, place of enumeration"
"POA (EN) and FMCF Family Composition by LFSF Labour Force Status of Parents/Partners in Families by TENLLD Tenure and Landlord Type"
"Counting: Family Records"

Filters:
"Default Summation","Family Records"
"TENLLD Tenure and Landlord Type","Rented: Real estate agent"

,"LFSF Labour Force Status of Parents/Partners in Families","Couple family: Both employed, worked full-time","One parent family: Not in the labour force","Total",
"POA (EN)","FMCF Family Composition",
"2000, NSW","Couple family with no children",7,0,7,
,"One parent family with children under 15",0,3,3,
"Total","Couple family with no children",7,0,7,
,"One parent family with children under 15",0,3,3,
"""


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body + FOOTER, encoding="utf-8")
    return str(path)


def test_column_variable_named_first_gets_its_own_label(tmp_path):
    path = write(tmp_path, "2021_TBP_POA_LFSP_SEXP_TENLLD-REA.csv", PERSONS)
    reader = tbr.TableBuilderReader(path, column_variable="LFSP")
    df = reader.df
    assert list(df.columns) == ["POA", "SEXP", "LFSP", "Persons"]
    assert set(df["SEXP"]) == {"Male", "Female"}
    assert set(df["LFSP"]) == {"Employed, worked full-time", "Not in the labour force"}
    assert list(reader.variables) == ["POA", "SEXP", "LFSP"]
    nilf = df[(df["SEXP"] == "Female") & (df["LFSP"] == "Not in the labour force")]
    assert nilf["Persons"].sum() == 11


def test_column_variable_named_last_is_unchanged(tmp_path):
    """Also pins the row-label fill: POA over two sexes is exactly half blank,
    which the old ">50% blank" fill rule skipped, losing every Female row."""
    path = write(tmp_path, "2021_TBP_POA_SEXP_LFSP_TENLLD-REA.csv", PERSONS)
    df = tbr.TableBuilderReader(path, column_variable="LFSP").df
    assert list(df.columns) == ["POA", "SEXP", "LFSP", "Persons"]
    assert set(df["SEXP"]) == {"Male", "Female"}


def test_families_dataset_counts_families(tmp_path):
    path = write(tmp_path, "2021_TBF_POA_FMCF(2 digit)_LFSF_TENLLD-REA.csv", FAMILIES)
    reader = tbr.TableBuilderReader(path, column_variable="LFSF")
    assert (reader.count, reader.count_code) == ("Families", "TBF")
    assert list(reader.df.columns) == ["POA", "FMCF", "LFSF", "Families"]
    assert reader.df["Families"].sum() == 10


# Frame length and count sum from Bond Loss's ColumnLastReader (nilf_tables.py) on
# the same files, 2026-11-08. The whole frames were also compared and are equal.
REAL_CASES = [
    ("2021_TBP_POA_LFSP_SEXP_AGE5P_TENLLD-REA.csv", "LFSP",
     ["POA", "SEXP", "AGE5P", "LFSP", "Persons"], 887_376, 4_665_630),
    ("2021_TBF_POA_FMCF(2 digit)_LFSF_TENLLD-REA.csv", "LFSF",
     ["POA", "FMCF", "LFSF", "Families"], 536_123, 2_007_623),
]


@pytest.mark.realdata
@pytest.mark.parametrize("name, column_variable, columns, rows, total", REAL_CASES)
def test_real_files_label_the_column_variable(
    name, column_variable, columns, rows, total
):
    reader = tbr.TableBuilderReader(rf"{RAW}\{name}", column_variable=column_variable)
    df = reader.df
    assert list(df.columns) == columns
    assert len(df) == rows
    assert df[columns[-1]].sum() == total
    # The labour force label is on the labour force column, not a row variable.
    assert df[column_variable].str.contains("employed", case=False).any()
    assert not df[columns[1]].str.contains("employed", case=False).any()
    if "SEXP" in df:
        assert set(df["SEXP"]) == {"Male", "Female"}
