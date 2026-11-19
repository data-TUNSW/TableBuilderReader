"""Build the combined weightings files.

Run as a script to regenerate everything:

    python -m abs_methods.build_weightings

See docs/superpowers/specs/2026-09-24-combined-geographic-weightings-design.md
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from abs_methods import geographic_weightings as gw
from abs_methods import table_builder_reader as tbr
from abs_methods import weightings_registry as reg


def read_universe(
    paths: list[Path], count_type: str, geog1: str, geog2: str
) -> pd.DataFrame:
    """Read a universe's chunks and concatenate them into one national frame.

    Concatenation happens before any totals are computed. Weighting a chunk on
    its own would give a geography that spans two chunks a fraction summing to
    1 within each of them, and to more than 1 nationally, with counts that look
    entirely correct.
    """
    frames = [
        tbr.TableBuilderReader(str(path), add_filter={count_type: "> 0"}).df
        for path in paths
    ]
    combined = pd.concat(frames, ignore_index=True)

    # The TableBuilder column name is the geography token, so it is both the
    # column to fix and the key into GEOGRAPHY_CODE_WIDTHS.
    for column in (geog1, geog2):
        combined[column] = reg.normalise_geography_column(
            combined[column], column
        )

    # A row whose geography the reader could not resolve is not in either
    # geography, so it belongs in neither total. The 2011 extracts carry an
    # "Unclassified (Aust.)" postal area with a few dozen people in it; the
    # reader's four-digit POA extraction turns that label into NA, and an NA
    # key would otherwise reach the totals and fail the QA checks.
    return combined.dropna(subset=[geog1, geog2]).reset_index(drop=True)


def build_pair(
    pair: reg.GeographyPair,
    raw_dir: Path | None = None,
    out_dir: Path = reg.WEIGHTINGS_DIR,
    write: bool = True,
) -> pd.DataFrame | None:
    """Build one geography pair's combined weightings file.

    `raw_dir` defaults to the pair's own Census raw folder (`pair.raw_dir`);
    pass one to override it, e.g. in tests.

    Returns None when none of the pair's universes has an extract yet.
    """
    if raw_dir is None:
        raw_dir = pair.raw_dir
    frames: dict[str, pd.DataFrame] = {}
    for universe in gw.UNIVERSES:
        pattern = pair.sources.get(universe.key)
        if pattern is None:
            continue
        paths = reg.resolve_sources(pattern, raw_dir)
        if not paths:
            continue

        extract = read_universe(paths, universe.count_type, pair.geog1, pair.geog2)
        frames[universe.key] = gw.calculate_geog_weighting(
            extract,
            geog1=pair.geog1,
            geog1_name=pair.geog1_name,
            geog2=pair.geog2,
            geog2_name=pair.geog2_name,
            count_type=universe.count_type,
            count_name=universe.count_name,
            add_shapefile_variables=False,
        )

    if not frames:
        return None

    combined = gw.combine_universes(frames, pair.geog1_name, pair.geog2_name)
    gw.check_weightings(
        combined,
        pair.geog1_name,
        pair.geog2_name,
        [u for u in gw.UNIVERSES if u.key in frames],
    )

    for shapefile, name_column, on in (
        (pair.geog1_shapefile, pair.geog1_shapefile_name, pair.geog1_name),
        (pair.geog2_shapefile, pair.geog2_shapefile_name, pair.geog2_name),
    ):
        if shapefile is None:
            # No shapefile configured for this geography: deliberate, silent.
            continue
        if not Path(shapefile).exists():
            # A shapefile WAS configured but its path is missing. That
            # tolerance exists so a pair can be built ahead of its shapefile
            # arriving, but it equally swallows a typo'd path with no other
            # signal - the file just comes out with no attribute columns.
            # Say so.
            print(
                f"{pair.key}: shapefile configured for {on} not found, "
                f"skipping attribute columns: {shapefile}"
            )
            continue
        attributes = gw.load_shapefile_attributes(shapefile, name_column)
        combined = gw.attach_shapefile_attributes(
            combined, attributes, on=on, name_column=name_column
        )

    if write:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        combined.to_csv(Path(out_dir) / pair.output_filename, index=False)
    return combined


def build_all(
    raw_dir: Path | None = None,
    out_dir: Path = reg.WEIGHTINGS_DIR,
    write: bool = True,
    mesh_blocks: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build every registered pair, then write the README.

    Each pair reads from its own Census raw folder unless `raw_dir` overrides
    it for all of them. The mesh block pairs (reg.MESH_BLOCK_PAIRS) follow
    the TableBuilder ones; they read boundaries and take several minutes,
    and `mesh_blocks=False` skips them.
    """
    built: dict[str, pd.DataFrame] = {}
    for pair in reg.PAIRS:
        combined = build_pair(pair, raw_dir=raw_dir, out_dir=out_dir, write=write)
        if combined is None:
            print(f"{pair.key}: no extracts yet, skipped")
            continue
        built[pair.key] = combined
        print(f"{pair.key}: {len(combined):,} rows")

    if mesh_blocks:
        # Imported here: mesh_block_weightings imports this module.
        from abs_methods import mesh_block_weightings as mbw

        for mb_pair in reg.MESH_BLOCK_PAIRS:
            combined = mbw.build_mesh_block_pair(
                mb_pair, out_dir=out_dir, write=write
            )
            built[mb_pair.key] = combined
            print(f"{mb_pair.key}: {len(combined):,} rows (mesh blocks)")

    if write and built:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "README.md").write_text(
            render_readme(built), encoding="utf-8"
        )
    return built


def render_readme(built: dict[str, pd.DataFrame]) -> str:
    """Describe what was built. Generated; do not edit by hand."""
    today = dt.date.today().isoformat()
    lines = [
        "# Geographic Weightings",
        "",
        f"Generated {today} by `abs_methods.build_weightings.build_all()`.",
        "Do not edit by hand.",
        "",
        "Each file is one geography pair, named",
        "`<geog1><edition>_<geog2><edition>_C<census>_Weightings.csv`:",
        "`POA2021_LGA2025_C2021` is 2021 postal areas against 2025 LGAs,",
        "weighted by 2021 Census counts. The edition is the year of the ASGS",
        "boundary release. LGA, SED and CED boundaries are re-released as",
        "often as yearly and POA, SAL and the SAs every five years, so files",
        "on different editions sit side by side; pick the editions that match",
        "your data. Each geography column carries its edition too, e.g.",
        "`Postcode (2021)`.",
        "",
        "Each universe contributes five",
        "columns: the count, both geographies' totals, and both fractions.",
        "",
        "`<geog1> Fraction <universe>` is the share of geog1 falling in geog2:",
        "use it to push geog1 data up to geog2. `<geog2> Fraction <universe>`",
        "is the share of geog2 made up by geog1: use it to push geog2 data",
        "down to geog1.",
        "",
        "A fraction is NaN where its total is zero, meaning that geography has",
        "no presence in that universe at all. Check the total before using a",
        "fraction.",
        "",
        "## Files",
        "",
        "| File | Rows | Universes |",
        "|---|---|---|",
    ]
    names = list(dict.fromkeys(u.count_name for u in _all_universes()))
    for key, df in built.items():
        universes = [name for name in names if name in df.columns]
        lines.append(
            f"| `{key}_Weightings.csv` | {len(df):,} | {', '.join(universes)} |"
        )
    lines.append("")

    mesh_block_keys = [k for k in built if k in reg.MESH_BLOCK_PAIRS_BY_KEY]
    if mesh_block_keys:
        lines.extend(_mesh_block_method(mesh_block_keys))
    return "\n".join(lines)


def _all_universes() -> list[gw.Universe]:
    """TableBuilder universes, then the mesh block ones."""
    # Imported here: mesh_block_weightings imports this module.
    from abs_methods import mesh_block_weightings as mbw

    return [*gw.UNIVERSES, *mbw.MESH_BLOCK_UNIVERSES]


def _mesh_block_method(keys: list[str]) -> list[str]:
    """README section on the files built from mesh blocks."""
    files = ", ".join(f"`{key}_Weightings.csv`" for key in keys)
    return [
        "## Files built from mesh blocks",
        "",
        f"{files} cannot come from TableBuilder: no Census was released on",
        "both of its geographies. It is assembled from 2026 mesh blocks:",
        "",
        "- Postal area: the ABS POA 2026 allocation file (ASGS Edition 4).",
        "- LGA 2011: spatial overlay of the 2026 mesh blocks on the ASGS 2011",
        "  LGA boundaries. Each mesh block goes whole to the LGA holding the",
        "  largest share of its area. Both layers are reprojected to GDA2020",
        "  Australian Albers, which also applies the GDA94 to GDA2020 shift",
        "  of about 1.5-1.8 m. The 2011 file has no polygons for",
        "  Unincorporated areas (all of the ACT, the NSW far west, Lord Howe",
        "  Island): the part of a mesh block inside a 2011 state but outside",
        "  its councils counts towards that state's Unincorporated LGA. LGA",
        "  labels match TableBuilder's 2011 labels (`Gosford (C)`,",
        "  `Campbelltown (C) (NSW)`), and `LGA_CODE11` carries the code.",
        "- Counts: 2021 Census mesh block counts, carried onto 2026 mesh",
        "  blocks by the ABS MB 2021 to MB 2026 correspondence. `Persons` is",
        "  persons usually resident. `Dwellings` is all dwellings, occupied",
        "  and unoccupied; it is not households.",
        "- `Rented Households (Estimated)` is not a Census count at this",
        "  level. Each 2021 SA1's rented households (TableBuilder) are spread",
        "  over its mesh blocks in proportion to their dwellings.",
        "",
        "Persons with no usual address, and mesh blocks with no postal area,",
        "are left out, as in the TableBuilder files. Counts are fractional",
        "where the correspondence splits a 2021 mesh block.",
        "",
    ]


if __name__ == "__main__":
    build_all()
