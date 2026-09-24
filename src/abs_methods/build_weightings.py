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
    return combined


def build_pair(
    pair: reg.GeographyPair,
    raw_dir: Path = reg.RAW_DIR,
    out_dir: Path = reg.WEIGHTINGS_DIR,
    write: bool = True,
) -> pd.DataFrame | None:
    """Build one geography pair's combined weightings file.

    Returns None when none of the pair's universes has an extract yet.
    """
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
    raw_dir: Path = reg.RAW_DIR,
    out_dir: Path = reg.WEIGHTINGS_DIR,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build every registered pair, then write the README."""
    built: dict[str, pd.DataFrame] = {}
    for pair in reg.PAIRS:
        combined = build_pair(pair, raw_dir=raw_dir, out_dir=out_dir, write=write)
        if combined is None:
            print(f"{pair.key}: no extracts yet, skipped")
            continue
        built[pair.key] = combined
        print(f"{pair.key}: {len(combined):,} rows")

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
        "Each file is one geography pair. Each universe contributes five",
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
    for key, df in built.items():
        universes = [
            u.count_name for u in gw.UNIVERSES if u.count_name in df.columns
        ]
        lines.append(
            f"| `{key}_Weightings.csv` | {len(df):,} | {', '.join(universes)} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    build_all()
