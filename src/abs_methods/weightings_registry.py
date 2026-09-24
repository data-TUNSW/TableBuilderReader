"""Where the weightings inputs and outputs live, and the pairs to build.

See docs/superpowers/specs/2026-09-24-combined-geographic-weightings-design.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

ABS_ROOT = Path("G:/Shared drives/Data/ABS")
RAW_DIR = ABS_ROOT / "2021 Census" / "raw"
GEOGRAPHY_DIR = ABS_ROOT / "Geography"

#: Output lives beside processed/, never inside it, so that a project globbing
#: processed/*.csv cannot pick up a combined file by accident.
WEIGHTINGS_DIR = GEOGRAPHY_DIR / "weightings"

SHAPEFILE_DIRS = {
    2021: GEOGRAPHY_DIR / "2021 shapefiles",
    2025: GEOGRAPHY_DIR / "2025 shapefiles",
}

#: Geographies whose codes are fixed-width numeric strings. Everything else is
#: a name and is only stripped.
GEOGRAPHY_CODE_WIDTHS: dict[str, int] = {"POA": 4, "SA1": 11}


def resolve_sources(
    patterns: str | Sequence[str], raw_dir: Path = RAW_DIR
) -> list[Path]:
    """Resolve a universe's source glob(s) to its chunk paths, sorted.

    ``patterns`` is either a single glob pattern or a sequence of them. A
    sequence is resolved pattern by pattern and the results combined into a
    sorted, deduplicated union - needed for universes registered as an
    explicit list of per-state filenames rather than one wildcard, because a
    single glob can be too greedy: ``Path.glob`` lets ``*`` match across
    underscores, so ``2021_TBP_POA_SAL-*.csv`` also matches per-tenure-type
    chunks like ``2021_TBP_POA_SAL-NSW_TEND-Rented.csv`` that belong to a
    different universe. Concatenating those in would silently inflate a
    universe's totals without tripping the fractions-sum-to-1 QA check.

    Returns an empty list when nothing matches (an extract not yet pulled);
    the caller treats that universe as absent rather than failing.
    """
    raw_dir = Path(raw_dir)
    if isinstance(patterns, str):
        patterns = [patterns]

    found: set[Path] = set()
    for pattern in patterns:
        found.update(raw_dir.glob(pattern))

    return sorted(found)


def normalise_geography_column(series: pd.Series, geography: str) -> pd.Series:
    """Coerce a geography column to the string form joins depend on.

    Postcode 0800 read from CSV as integer 800 would fail to match every
    shapefile row and every other universe, without raising anything.
    """
    values = series.astype("string").str.strip()
    width = GEOGRAPHY_CODE_WIDTHS.get(geography)
    if width is not None:
        values = values.str.zfill(width)
    return values
