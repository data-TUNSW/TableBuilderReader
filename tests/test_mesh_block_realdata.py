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
def labels_2011():
    """Every LGA label in the TableBuilder 2011 file."""
    df = _read(reg.WEIGHTINGS_DIR / TABLEBUILDER_2011.output_filename)
    return set(df["Local Government Area (2011)"])


def test_file_passes_its_own_qa(built):
    gw.check_weightings(built, POA, LGA, mbw.MESH_BLOCK_UNIVERSES)


def test_all_153_nsw_labels_are_present_and_match_the_2011_file(
    built, labels_2011
):
    """152 councils plus Unincorporated NSW, spelt as TableBuilder spells them.

    The 2011 file has no code column, so NSW is told apart here by code.
    """
    nsw = set(built.loc[built["LGA_CODE11"].astype(str).str.startswith("1"), LGA])
    assert len(nsw) == 153
    assert "Unincorporated NSW" in nsw
    assert "Campbelltown (C) (NSW)" in nsw
    assert nsw <= labels_2011


def test_every_label_has_one_code(built):
    assert built.groupby(LGA)["LGA_CODE11"].nunique().eq(1).all()


def test_pre_merger_councils_split_shared_postcodes(built):
    pairs = set(zip(built[POA], built[LGA]))
    assert ("2250", "Gosford (C)") in pairs
    assert ("2259", "Wyong (A)") in pairs
    assert ("2204", "Marrickville (A)") in pairs
    assert ("2620", "Queanbeyan (C)") in pairs
    assert ("2620", "Palerang (A)") in pairs
