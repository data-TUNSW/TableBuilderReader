"""Where the weightings inputs and outputs live, and the pairs to build.

See docs/superpowers/specs/2026-09-24-combined-geographic-weightings-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

ABS_ROOT = Path("G:/Shared drives/Data/ABS")


def raw_dir_for(census: int) -> Path:
    """The folder a Census's TableBuilder extracts live in."""
    return ABS_ROOT / f"{census} Census" / "raw"


RAW_DIR = raw_dir_for(2021)
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


#: The default Census for a pair, and the one whose extracts are in RAW_DIR.
#: A pair on another Census reads from raw_dir_for(pair.census) instead.
CENSUS_YEAR = 2021


@dataclass(frozen=True)
class GeographyPair:
    """One geography pair and where its universes come from.

    A pair is identified by three versions, because any one of them changing
    changes the fractions: each geography's ASGS edition (the year of the
    boundary release, e.g. POA 2021, LGA 2025) and the Census the counts come
    from. All three go into the key and the output filename, so a file built
    on new boundaries or a new Census sits beside the old one instead of
    silently replacing it. LGA, SED and CED are re-released as often as
    yearly; POA, SAL and the SAs every five years.

    Each geography's column name must end with its edition in brackets, e.g.
    ``Postcode (2021)``, so the year travels with the column into whatever
    joins against it.
    """

    geog1: str
    geog1_edition: int
    geog1_name: str
    geog2: str
    geog2_edition: int
    geog2_name: str
    sources: dict[str, str | list[str]]
    census: int = CENSUS_YEAR
    geog1_shapefile: Path | None = None
    geog1_shapefile_name: str | None = None
    geog2_shapefile: Path | None = None
    geog2_shapefile_name: str | None = None

    def __post_init__(self) -> None:
        for name, edition in (
            (self.geog1_name, self.geog1_edition),
            (self.geog2_name, self.geog2_edition),
        ):
            if not name.endswith(f"({edition})"):
                raise ValueError(
                    f"column name {name!r} must end with its edition "
                    f"'({edition})'"
                )

    @property
    def key(self) -> str:
        """e.g. ``POA2021_LGA2025_C2021``."""
        return (
            f"{self.geog1}{self.geog1_edition}_"
            f"{self.geog2}{self.geog2_edition}_C{self.census}"
        )

    @property
    def output_filename(self) -> str:
        return f"{self.key}_Weightings.csv"

    @property
    def raw_dir(self) -> Path:
        """Where this pair's extracts live: its own Census's raw folder."""
        return raw_dir_for(self.census)


_POA_SHAPEFILE = SHAPEFILE_DIRS[2021] / "POA_2021_AUST_GDA2020.shp"

#: States/territories the POA x SAL extract is pulled per, because that
#: cross-tab exceeds TableBuilder's 40,000,000 cell limit at the national
#: level.
CHUNK_STATES = ("NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT", "OT")


def _poa_sources(
    geog2_token: str, chunked: bool = False, census: int = CENSUS_YEAR
) -> dict[str, str | list[str]]:
    """The six universe globs for a POA-by-geog2 cross-tab.

    `chunked` appends the state wildcard used when a cross-tab exceeds
    TableBuilder's 40,000,000 cell limit and has to be pulled per state.
    `census` is the year prefix on every extract filename.
    """
    token = f"{geog2_token}-*" if chunked else geog2_token
    c = census
    sources: dict[str, str | list[str]] = {
        "all_households": f"{c}_TBD_POA_{token}_DWTD-Occupied.csv",
        "all_persons": f"{c}_TBP_POA_{token}.csv",
        "rented_households": f"{c}_TBD_POA_{token}_TEND-Rented.csv",
        "renters_people": f"{c}_TBP_POA_{token}_TEND-Rented.csv",
        "private_rented_households": (
            f"{c}_TBD_POA_{token}_TENLLD-Rented Real estate agent.csv"
        ),
        "private_renters_people": (
            f"{c}_TBP_POA_{token}_TENLLD-Rented Real estate agent.csv"
        ),
    }
    if chunked:
        # all_persons cannot use the `{geog2_token}-*` glob like the other
        # five universes: unlike them, its filename has nothing after the
        # state code to disambiguate it, so `Path.glob`'s `*` (which crosses
        # underscores) also sweeps in the per-tenure chunks of the other
        # person-basis universes, e.g. `2021_TBP_POA_SAL-NSW_TEND-Rented.csv`
        # and `...SAL-NSW_TENLLD-Rented Real estate agent.csv`. That would
        # silently inflate all_persons' totals without tripping the
        # fractions-sum-to-1 QA check, since the concatenated frame's
        # fractions still sum to 1. List the nine per-state files explicitly
        # instead - do not "simplify" this back to a glob.
        sources["all_persons"] = [
            f"{c}_TBP_POA_{geog2_token}-{state}.csv" for state in CHUNK_STATES
        ]
    return sources


PAIRS: tuple[GeographyPair, ...] = (
    GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="LGA",
        geog2_edition=2025,
        geog2_name="Local Government Area (2025)",
        sources=_poa_sources("LGA(2025)"),
        geog1_shapefile=_POA_SHAPEFILE,
        geog1_shapefile_name="POA_NAME21",
        geog2_shapefile=SHAPEFILE_DIRS[2025] / "LGA_2025_AUST_GDA2020.shp",
        geog2_shapefile_name="LGA_NAME25",
    ),
    GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="SED",
        geog2_edition=2025,
        geog2_name="State Electorate (2025)",
        sources=_poa_sources("SED(2025)"),
        geog1_shapefile=_POA_SHAPEFILE,
        geog1_shapefile_name="POA_NAME21",
        geog2_shapefile=SHAPEFILE_DIRS[2025] / "SED_2025_AUST_GDA2020.shp",
        geog2_shapefile_name="SED_NAME25",
    ),
    GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="CED",
        geog2_edition=2025,
        geog2_name="Federal Electorate (2025)",
        sources=_poa_sources("CED(2025)"),
        geog1_shapefile=_POA_SHAPEFILE,
        geog1_shapefile_name="POA_NAME21",
        geog2_shapefile=SHAPEFILE_DIRS[2025] / "CED_2025_AUST_GDA2020.shp",
        geog2_shapefile_name="CED_NAME25",
    ),
    GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="SA3",
        geog2_edition=2021,
        geog2_name="SA3 (2021)",
        sources=_poa_sources("SA3"),
        geog1_shapefile=_POA_SHAPEFILE,
        geog1_shapefile_name="POA_NAME21",
        geog2_shapefile=SHAPEFILE_DIRS[2021] / "SA3_2021_AUST_GDA2020.shp",
        geog2_shapefile_name="SA3_NAME21",
    ),
    # GCCSA is an ASGS main-structure region, released with POA and the SAs
    # every five years, so its edition matches the Census it was pulled from.
    GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="GCCSA",
        geog2_edition=2021,
        geog2_name="GCCSA (2021)",
        sources=_poa_sources("GCCSA"),
        geog1_shapefile=_POA_SHAPEFILE,
        geog1_shapefile_name="POA_NAME21",
        geog2_shapefile=SHAPEFILE_DIRS[2021] / "GCCSA_2021_AUST_GDA2020.shp",
        # The ABS abbreviates GCCSA to GCC in its shapefile columns.
        geog2_shapefile_name="GCC_NAME21",
    ),
    # 2016 Census counts on ASGS 2016 boundaries. No 2016 POA or GCCSA
    # shapefiles are on G:, so this file carries no attribute columns.
    GeographyPair(
        geog1="POA",
        geog1_edition=2016,
        geog1_name="Postcode (2016)",
        geog2="GCCSA",
        geog2_edition=2016,
        geog2_name="GCCSA (2016)",
        sources=_poa_sources("GCCSA", census=2016),
        census=2016,
    ),
    GeographyPair(
        geog1="POA",
        geog1_edition=2021,
        geog1_name="Postcode (2021)",
        geog2="SAL",
        geog2_edition=2021,
        geog2_name="Suburb (2021)",
        sources=_poa_sources("SAL", chunked=True),
        geog1_shapefile=_POA_SHAPEFILE,
        geog1_shapefile_name="POA_NAME21",
        geog2_shapefile=SHAPEFILE_DIRS[2021] / "SAL_2021_AUST_GDA2020.shp",
        geog2_shapefile_name="SAL_NAME21",
    ),
    GeographyPair(
        geog1="SA1",
        geog1_edition=2021,
        geog1_name="SA1 (2021)",
        geog2="LGA",
        geog2_edition=2021,
        geog2_name="Local Government Area (2021)",
        sources={"rented_households": "2021_TBD_SA1_LGA_TEND-Rented.csv"},
        geog1_shapefile=SHAPEFILE_DIRS[2021] / "SA1_2021_AUST_GDA2020.shp",
        geog1_shapefile_name="SA1_CODE21",
        geog2_shapefile=SHAPEFILE_DIRS[2021] / "LGA_2021_AUST_GDA2020.shp",
        geog2_shapefile_name="LGA_NAME21",
    ),
)

PAIRS_BY_KEY: dict[str, GeographyPair] = {p.key: p for p in PAIRS}
