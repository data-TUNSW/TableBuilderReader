"""set_count reads the dataset code as a whole file-name token.

It used to test ``"TBP" in file_name`` before ``"TBP15"``, so every TBP15
file got the TBP count ("Persons", not "Persons over 15"), and a substring
match anywhere in the name could pick the wrong dataset.
"""

import pytest

from abs_methods import table_builder_reader as tbr


@pytest.mark.parametrize(
    "name, count, code",
    [
        ("2021_TBD_POA_HCFMD.csv", "Dwellings", "TBD"),
        ("2021_TBP_POA_SEXP.csv", "Persons", "TBP"),
        ("2021_TBP15_POA_SEXP.csv", "Persons over 15", "TBP15"),
        ("TBP15_POA_SEXP.csv", "Persons over 15", "TBP15"),
        ("2021_TBF_POA_FMCF.csv", "Families", "TBF"),
        # A filter category that happens to contain a code is not the dataset.
        ("2021_TBP_POA_DWTD-TBD.csv", "Persons", "TBP"),
    ],
)
def test_set_count_reads_the_dataset_token(name, count, code):
    reader = tbr.TableBuilderReader(name, infer_from_file_name=False)
    reader.set_count()
    assert (reader.count, reader.count_code) == (count, code)
