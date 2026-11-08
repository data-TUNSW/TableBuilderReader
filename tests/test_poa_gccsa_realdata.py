"""POA x GCCSA built from the real extracts, in memory (nothing is written).

Needs G:. Deselected by default; run with ``python -m pytest -m realdata``.
"""

import pandas as pd
import pytest

from abs_methods import build_weightings as bw
from abs_methods import geographic_weightings as gw
from abs_methods import table_builder_reader as tbr
from abs_methods import weightings_registry as reg

pytestmark = pytest.mark.realdata

KEYS = ["POA2021_GCCSA2021_C2021", "POA2016_GCCSA2016_C2016"]


@pytest.fixture(scope="module")
def built():
    frames = {}
    for key in KEYS:
        pair = reg.PAIRS_BY_KEY[key]
        if not pair.raw_dir.exists():
            pytest.skip(f"missing {pair.raw_dir}")
        frames[key] = bw.build_pair(pair, write=False)
    return frames


@pytest.mark.parametrize("key", KEYS)
def test_every_universe_is_built(built, key):
    df = built[key]
    for universe in gw.UNIVERSES:
        assert universe.count_name in df.columns, universe.count_name


@pytest.mark.parametrize("key", KEYS)
def test_gccsa_totals_add_up_to_the_raw_national_total(built, key):
    """Nothing is dropped or double counted between the extract and the file."""
    pair = reg.PAIRS_BY_KEY[key]
    df = built[key]
    for universe in gw.UNIVERSES:
        paths = reg.resolve_sources(pair.sources[universe.key], pair.raw_dir)
        raw = pd.concat([tbr.TableBuilderReader(str(p)).df for p in paths])
        total_column = f"{pair.geog2_name} Total {universe.count_name}"
        gccsa_totals = df.drop_duplicates(pair.geog2_name)[total_column].sum()
        assert gccsa_totals == raw[universe.count_type].sum(), universe.count_name


@pytest.mark.parametrize(
    "key,low,high",
    [
        ("POA2021_GCCSA2021_C2021", 5_000_000, 5_500_000),
        ("POA2016_GCCSA2016_C2016", 4_700_000, 5_100_000),
    ],
)
def test_greater_sydney_persons_is_plausible(built, key, low, high):
    pair = reg.PAIRS_BY_KEY[key]
    df = built[key]
    totals = df.drop_duplicates(pair.geog2_name).set_index(pair.geog2_name)[
        f"{pair.geog2_name} Total Persons"
    ]
    assert low < totals["Greater Sydney"] < high
    assert totals["Greater Sydney"] > totals["Rest of NSW"]


@pytest.mark.parametrize("key", KEYS)
def test_cross_border_postcode_2620_splits_between_act_and_rest_of_nsw(built, key):
    pair = reg.PAIRS_BY_KEY[key]
    df = built[key]
    rows = df[(df[pair.geog1_name] == "2620") & (df["Persons"] > 0)]
    assert set(rows[pair.geog2_name]) >= {
        "Australian Capital Territory",
        "Rest of NSW",
    }
    assert rows[f"{pair.geog1_name} Fraction Persons"].sum() == pytest.approx(1.0)
