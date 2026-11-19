"""Checks on the written POA2026_LGA2011_C2021 file against the 2011 one.

Reads the files in the weightings folder rather than rebuilding (the overlay
takes several minutes). Needs G:. Deselected by default; run with
``python -m pytest -m realdata``.
"""

import pandas as pd
import pytest

from abs_methods import geographic_weightings as gw
from abs_methods import mesh_block_weightings as mbw
from abs_methods import weightings_registry as reg

pytestmark = pytest.mark.realdata

PAIR = reg.MESH_BLOCK_PAIRS_BY_KEY["POA2026_LGA2011_C2021"]
POA, LGA = PAIR.geog1_name, PAIR.geog2_name
TABLEBUILDER_2011 = reg.PAIRS_BY_KEY["POA2011_LGA2011_C2011"]


def _read(path):
    if not path.exists():
        pytest.skip(f"missing {path}")
    return pd.read_csv(path, dtype={POA: str, "Postcode (2011)": str})


@pytest.fixture(scope="module")
def built():
    return _read(reg.WEIGHTINGS_DIR / PAIR.output_filename)


@pytest.fixture(scope="module")
def nsw_2011_labels():
    """The 153 NSW labels in TableBuilder order, Albury to Unincorporated."""
    df = _read(reg.WEIGHTINGS_DIR / TABLEBUILDER_2011.output_filename)
    labels = list(dict.fromkeys(df["Local Government Area (2011)"]))
    start = labels.index("Albury (C)")
    end = labels.index("Unincorporated NSW")
    return labels[start : end + 1]


def test_file_passes_its_own_qa(built):
    gw.check_weightings(built, POA, LGA, mbw.MESH_BLOCK_UNIVERSES)


def test_every_nsw_2011_council_is_present_under_its_tablebuilder_label(
    built, nsw_2011_labels
):
    assert len(nsw_2011_labels) == 153
    assert set(nsw_2011_labels) <= set(built[LGA])


def test_every_nsw_label_carries_a_nsw_code(built, nsw_2011_labels):
    nsw = built[built[LGA].isin(nsw_2011_labels)]
    assert nsw["LGA_CODE11"].astype(str).str.startswith("1").all()
    assert nsw.groupby(LGA)["LGA_CODE11"].nunique().eq(1).all()


def test_pre_merger_councils_split_shared_postcodes(built):
    pairs = set(zip(built[POA], built[LGA]))
    assert ("2250", "Gosford (C)") in pairs
    assert ("2259", "Wyong (A)") in pairs
    assert ("2204", "Marrickville (A)") in pairs
    assert ("2620", "Queanbeyan (C)") in pairs
    assert ("2620", "Palerang (A)") in pairs
