"""Combined geographic weightings.

One weightings file per geography pair, carrying every available universe as
columns. See
docs/superpowers/specs/2026-09-24-combined-geographic-weightings-design.md

`calculate_geog_weighting` was moved here verbatim from the "Functions" cell of
src/geographic_weightings.ipynb, which is not tracked by git (*.ipynb is in
.gitignore). This module is the version-controlled record of that logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Universe:
    """One count universe: a TableBuilder extract with its own filter.

    key:        stable identifier used in the registry and in test fixtures
    count_name: the name the count takes in the output columns
    count_type: the count column as TableBuilder emits it
    basis:      "TBD" for dwelling records, "TBP" for person records
    """

    key: str
    count_name: str
    count_type: str
    basis: str


#: Fixed order. This is the order universe column blocks appear in the output,
#: so a file's column order is predictable from the universes it has.
UNIVERSES: tuple[Universe, ...] = (
    Universe("all_households", "Households", "Dwellings", "TBD"),
    Universe("all_persons", "Persons", "Persons", "TBP"),
    Universe("rented_households", "Rented Households", "Dwellings", "TBD"),
    Universe("renters_people", "Renters (People)", "Persons", "TBP"),
    Universe(
        "private_rented_households",
        "Private Rented Households",
        "Dwellings",
        "TBD",
    ),
    Universe(
        "private_renters_people",
        "Private Renters (People)",
        "Persons",
        "TBP",
    ),
)

UNIVERSES_BY_KEY: dict[str, Universe] = {u.key: u for u in UNIVERSES}


def universe_columns(
    universe: Universe, geog1_name: str, geog2_name: str
) -> list[str]:
    """The five output columns a universe contributes, in block order."""
    count = universe.count_name
    return [
        count,
        f"{geog1_name} Total {count}",
        f"{geog2_name} Total {count}",
        f"{geog1_name} Fraction {count}",
        f"{geog2_name} Fraction {count}",
    ]


def calculate_geog_weighting(
    df,
    geog1,
    geog1_name,
    geog2,
    geog2_name,
    count_type,
    count_name=None,
    add_shapefile_variables=True,
    add_geometry=False,
    geog1_shapefile_path=None,
    geog1_shapefile_name="POA_NAME21",
    geog2_shapefile_path=None,
    geog2_shapefile_name="SED_NAME25",
):
    """Weight one geography against another for a single count universe.

    Moved verbatim from src/geographic_weightings.ipynb. Behaviour is
    unchanged so that the files in Geography/processed/ remain reproducible.
    """
    count_col = count_type
    geog1_total_col = f"{geog1_name} Total {count_col}"
    geog2_total_col = f"{geog2_name} Total {count_col}"
    count_suffix = f"{count_col}"

    # Calculate totals for each geography
    geog1_totals = df.groupby(geog1)[count_col].sum().reset_index()
    geog1_totals = geog1_totals.rename(columns={count_col: geog1_total_col})
    geog2_totals = df.groupby(geog2)[count_col].sum().reset_index()
    geog2_totals = geog2_totals.rename(columns={count_col: geog2_total_col})

    # Merge totals back to main dataframe
    weighting = (
        df.merge(geog1_totals, on=geog1, how="left")
        .merge(geog2_totals, on=geog2, how="left")
        .rename(
            columns={
                count_col: count_suffix,
                geog1: geog1_name,
                geog2: geog2_name,
            }
        )
    )
    weighting[f"{geog1_name} Fraction"] = (
        weighting[count_suffix] / weighting[geog1_total_col]
    )
    weighting[f"{geog2_name} Fraction"] = (
        weighting[count_suffix] / weighting[geog2_total_col]
    )

    if add_shapefile_variables:
        for shapefile_path, shapefile_name, join_name in (
            (geog1_shapefile_path, geog1_shapefile_name, geog1_name),
            (geog2_shapefile_path, geog2_shapefile_name, geog2_name),
        ):
            if shapefile_path is None:
                continue
            geo = gpd.read_file(shapefile_path)
            geo = geo.drop(
                columns=[
                    "AUS_CODE21",
                    "AUS_NAME21",
                    "LOCI_URI21",
                    "SHAPE_Leng",
                    "SHAPE_Area",
                ],
                errors="ignore",
            )
            if "AREASQKM" in geo.columns:
                geo = geo.rename(
                    columns={"AREASQKM": f"{shapefile_name}_AREASQKM"}
                )
            if "AREASQKM21" in geo.columns:
                geo = geo.rename(
                    columns={"AREASQKM21": f"{shapefile_name}_AREASQKM21"}
                )
            if not add_geometry and "geometry" in geo.columns:
                geo = geo.drop(columns=["geometry"])
            else:
                geo = geo.rename(
                    columns={"geometry": f"{shapefile_name}_Geometry"}
                )
            if shapefile_name not in geo.columns:
                raise ValueError(
                    f"Shapefile {shapefile_path}\n"
                    + f"does not contain the specified name column: {shapefile_name}"
                )
            weighting = weighting.merge(
                geo,
                left_on=join_name,
                right_on=shapefile_name,
                how="left",
            )

    # Relabel the count columns, if the caller asked for a different name
    if count_name is not None:
        weighting = weighting.rename(
            columns={
                col: col.replace(count_col, count_name)
                for col in weighting.columns
                if count_col in col
            }
        )

    return weighting


#: Columns the ABS ships in its shapefiles that carry no analytical value.
SHAPEFILE_BOILERPLATE_COLUMNS = (
    "AUS_CODE21",
    "AUS_NAME21",
    "LOCI_URI21",
    "SHAPE_Leng",
    "SHAPE_Area",
)


def clean_shapefile_attributes(frame, name_column: str) -> pd.DataFrame:
    """Reduce a shapefile to the attribute columns worth keeping.

    Accepts a GeoDataFrame or a plain DataFrame. Drops geometry and the ABS
    boilerplate columns, prefixes the area column with the name column so two
    geographies' areas do not collide, and strips the name column so it can be
    merged against a TableBuilder geography name.
    """
    attributes = pd.DataFrame(frame).drop(
        columns=[*SHAPEFILE_BOILERPLATE_COLUMNS, "geometry"], errors="ignore"
    )

    for area_column in ("AREASQKM", "AREASQKM21"):
        if area_column in attributes.columns:
            attributes = attributes.rename(
                columns={area_column: f"{name_column}_{area_column}"}
            )

    if name_column not in attributes.columns:
        raise ValueError(
            f"Shapefile does not contain the specified name column: {name_column}"
        )

    attributes[name_column] = (
        attributes[name_column].astype("string").str.strip()
    )
    return attributes


def load_shapefile_attributes(path, name_column: str) -> pd.DataFrame:
    """Read a shapefile and reduce it to its attribute columns."""
    return clean_shapefile_attributes(gpd.read_file(path), name_column)


def attach_shapefile_attributes(
    df: pd.DataFrame, attributes: pd.DataFrame, on: str, name_column: str
) -> pd.DataFrame:
    """Left-merge shapefile attributes onto a weightings frame.

    Both sides of the key are stripped. An unstripped key is a silent failure:
    the merge succeeds and every attribute column comes back null.
    """
    left = df.copy()
    key = f"__{on}_key"
    left[key] = left[on].astype("string").str.strip()

    merged = left.merge(
        attributes, left_on=key, right_on=name_column, how="left"
    )
    return merged.drop(columns=[key, name_column])


def _universe_block(
    frame: pd.DataFrame, universe: Universe, geog1_name: str, geog2_name: str
) -> pd.DataFrame:
    """Select one universe's columns and suffix its fractions.

    `calculate_geog_weighting` emits bare "<geog> Fraction" names, which are the
    only names that collide between universes.
    """
    count = universe.count_name
    block = frame[
        [
            geog1_name,
            geog2_name,
            count,
            f"{geog1_name} Total {count}",
            f"{geog2_name} Total {count}",
            f"{geog1_name} Fraction",
            f"{geog2_name} Fraction",
        ]
    ].copy()
    return block.rename(
        columns={
            f"{geog1_name} Fraction": f"{geog1_name} Fraction {count}",
            f"{geog2_name} Fraction": f"{geog2_name} Fraction {count}",
        }
    )


def _apply_fill_rules(
    combined: pd.DataFrame,
    frame: pd.DataFrame,
    universe: Universe,
    geog1_name: str,
    geog2_name: str,
) -> pd.DataFrame:
    """Fill one universe's block for pairs the outer join left empty.

    Count fills to 0: the pair genuinely contributes nothing. Totals fill from
    the universe's own totals, so a pair missing from this universe still
    carries its geographies' real denominators. Fractions fill to 0 where the
    denominator is above zero, and stay NaN where it is zero, which is the
    difference between "contributes 0% of the LGA's renters" and "has no
    renters to contribute".
    """
    count = universe.count_name
    combined[count] = combined[count].fillna(0)

    for geog_name in (geog1_name, geog2_name):
        total_column = f"{geog_name} Total {count}"
        fraction_column = f"{geog_name} Fraction {count}"

        totals = frame.drop_duplicates(geog_name).set_index(geog_name)[
            total_column
        ]
        combined[total_column] = (
            combined[total_column]
            .fillna(combined[geog_name].map(totals))
            .fillna(0)
        )
        combined[fraction_column] = (
            combined[fraction_column]
            .fillna(0)
            .where(combined[total_column] > 0, other=np.nan)
        )

    return combined


def combine_universes(
    frames: dict[str, pd.DataFrame], geog1_name: str, geog2_name: str
) -> pd.DataFrame:
    """Outer-join every available universe onto one geography-pair frame.

    `frames` maps a universe key to the frame `calculate_geog_weighting`
    produced for it. Universes absent from `frames` contribute no columns.
    """
    present = [u for u in UNIVERSES if u.key in frames]
    if not present:
        raise ValueError("combine_universes needs at least one universe")

    combined = None
    for universe in present:
        block = _universe_block(
            frames[universe.key], universe, geog1_name, geog2_name
        )
        combined = (
            block
            if combined is None
            else combined.merge(block, on=[geog1_name, geog2_name], how="outer")
        )

    for universe in present:
        combined = _apply_fill_rules(
            combined, frames[universe.key], universe, geog1_name, geog2_name
        )

    ordered = [geog1_name, geog2_name]
    for universe in present:
        ordered.extend(universe_columns(universe, geog1_name, geog2_name))
    return combined[ordered]


def check_weightings(
    df: pd.DataFrame,
    geog1_name: str,
    geog2_name: str,
    universes,
    tolerance: float = 1e-9,
) -> None:
    """Raise if a combined weightings frame is internally inconsistent.

    Collects every problem rather than stopping at the first, so one run tells
    you everything that is wrong.
    """
    problems: list[str] = []

    duplicates = int(df.duplicated(subset=[geog1_name, geog2_name]).sum())
    if duplicates:
        problems.append(
            f"{duplicates} duplicate ({geog1_name}, {geog2_name}) rows; "
            "the most likely cause is overlapping extract chunks"
        )

    for universe in universes:
        count = universe.count_name
        if (df[count] < 0).any():
            problems.append(f"{count}: negative counts")

        for geog_name in (geog1_name, geog2_name):
            total_column = f"{geog_name} Total {count}"
            fraction_column = f"{geog_name} Fraction {count}"

            if (df[count] > df[total_column]).any():
                problems.append(f"{count}: exceeds {total_column}")

            fraction = df[fraction_column]
            if ((fraction < 0) | (fraction > 1)).any():
                problems.append(f"{fraction_column}: outside [0, 1]")

            # A NaN fraction is only valid where the geography's own total is
            # zero. skipna in the sum-to-1 check below would otherwise let a
            # stray NaN in a live geography pass unnoticed, so check for it
            # directly instead of relying on that summation as a side effect.
            nan_in_live = fraction.isna() & (df[total_column] > 0)
            if nan_in_live.any():
                problems.append(
                    f"{fraction_column}: {int(nan_in_live.sum())} NaN "
                    f"fractions where {total_column} is above zero"
                )

            # Only geographies that exist in this universe should sum to 1.
            # A geography with a zero total has NaN fractions by design.
            live = df.groupby(geog_name)[total_column].max() > 0
            sums = df.groupby(geog_name)[fraction_column].sum(min_count=1)
            live_sums = sums[live.reindex(sums.index, fill_value=False)]
            off = live_sums[~np.isclose(live_sums, 1.0, atol=tolerance)]
            if len(off):
                problems.append(
                    f"{fraction_column}: {len(off)} {geog_name} values do not "
                    f"sum to 1 (worst: {off.abs().sub(1).idxmax()})"
                )

    if problems:
        raise ValueError("Weightings QA failed:\n  " + "\n  ".join(problems))
