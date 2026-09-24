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
