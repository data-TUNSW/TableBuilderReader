"""Weightings built from mesh blocks instead of a TableBuilder cross-tab.

Every other weightings file comes from one Census cross-tab of two
geographies. That only works when one Census was released on both. No Census
crosses 2011 LGAs (the NSW councils before the May 2016 mergers) with 2026
postal areas, so `POA2026_LGA2011_C2021` is assembled from 2026 mesh blocks:

1. Each 2026 mesh block's postal area comes from the ABS POA 2026 allocation
   file (ASGS Edition 4).
2. Each 2026 mesh block's 2011 LGA comes from a spatial overlay against the
   ASGS 2011 LGA boundaries: the LGA holding the largest share of the mesh
   block's area. The 2011 boundaries are GDA94 and the 2026 ones GDA2020;
   both are reprojected to GDA2020 Australian Albers (EPSG:9473), which
   applies the GDA94-to-GDA2020 shift (about 1.5 to 1.8 m) and measures
   areas in square metres.
3. The counts are the 2021 Census mesh block counts (persons usually
   resident, and all dwellings), carried onto 2026 mesh blocks through the
   ABS MB 2021 to MB 2026 correspondence. Rented households are estimated:
   each 2021 SA1's rented households are spread over its mesh blocks in
   proportion to their dwellings.

The resulting mesh block table is aggregated to (POA, LGA) pairs and then goes
through the same `calculate_geog_weighting`, `combine_universes` and
`check_weightings` as the TableBuilder files.

The ASGS 2011 LGA file has no polygon for the "Unincorporated" areas (all of
the ACT, the NSW far west and Lord Howe Island, and so on): they are defined
as whatever part of a state no council covers. The overlay therefore gives
each mesh block a further candidate per 2011 state, its area inside that
state's 2011 boundary less the area inside that state's councils, labelled
with the state's Unincorporated LGA. Area outside every 2011 state (mostly
coastline drawn differently in 2011) is no candidate at all, so a coastal mesh
block is not pushed into an Unincorporated area by its sea frontage.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from abs_methods import build_weightings as bw
from abs_methods import geographic_weightings as gw
from abs_methods import weightings_registry as reg

#: GDA2020 / Australian Albers. Equal-area, so overlap areas compare fairly
#: anywhere in Australia.
EQUAL_AREA_CRS = "EPSG:9473"

PERSONS = gw.UNIVERSES_BY_KEY["all_persons"]

#: Mesh block counts give all dwellings, occupied and unoccupied (bar
#: unoccupied ones in caravan parks, marinas and manufactured home estates),
#: not households, so the universe is named for what it counts.
DWELLINGS = gw.Universe("all_dwellings", "Dwellings", "Dwellings", "MB")

#: Not a Census count at mesh block level: see estimate_rented_households.
RENTED_ESTIMATED = gw.Universe(
    "rented_households_estimated",
    "Rented Households (Estimated)",
    "Rented Households (Estimated)",
    "MB",
)

#: Fixed order, as gw.UNIVERSES is for the TableBuilder files.
MESH_BLOCK_UNIVERSES: tuple[gw.Universe, ...] = (
    PERSONS,
    DWELLINGS,
    RENTED_ESTIMATED,
)

#: How TableBuilder 2011 tells apart LGA names that repeat across states,
#: e.g. "Campbelltown (C) (NSW)" and "Campbelltown (C) (SA)".
STATE_SUFFIXES = {
    "1": "NSW",
    "2": "Vic.",
    "3": "Qld",
    "4": "SA",
    "5": "WA",
    "6": "Tas.",
    "7": "NT",
    "8": "ACT",
    "9": "OT",
}


# ---------------------------------------------------------------- counts


def read_mesh_block_counts(path: Path) -> pd.DataFrame:
    """Read the ABS "Mesh Block Counts, 2021" workbook into one frame.

    The workbook has one or two sheets per state, each with six title rows
    above the header and a confidentiality footer below the data. Returns
    MB_CODE_2021, Dwellings and Persons, with the no-usual-address mesh
    blocks' blank dwelling count read as 0.
    """
    with pd.ExcelFile(path) as workbook:
        frames = [
            workbook.parse(name, header=6, dtype={"MB_CODE_2021": str})
            for name in workbook.sheet_names
            if name.startswith("Table")
        ]
    counts = pd.concat(frames, ignore_index=True)
    counts = counts[
        counts["MB_CODE_2021"].str.fullmatch(r"\d{11}", na=False)
    ]
    counts = counts.rename(columns={"Dwelling": "Dwellings", "Person": "Persons"})
    counts[["Dwellings", "Persons"]] = counts[["Dwellings", "Persons"]].fillna(0)
    if counts["MB_CODE_2021"].duplicated().any():
        raise ValueError(f"{path}: a mesh block appears more than once")
    return counts[["MB_CODE_2021", "Dwellings", "Persons"]].reset_index(
        drop=True
    )


def estimate_rented_households(
    mb_counts: pd.DataFrame,
    mb_to_sa1: pd.DataFrame,
    sa1_rented: pd.Series,
) -> pd.DataFrame:
    """Spread each SA1's rented households over its mesh blocks by dwellings.

    Census rented-household counts stop at SA1. Each mesh block gets its SA1's
    rented households times its share of the SA1's dwellings, which is the
    same as applying the SA1's rented share (rented households over all
    dwellings, occupied or not) to the mesh block's dwellings. Within an SA1
    the estimate therefore sums back to the Census count exactly, except
    where the SA1 has rented households but no mesh block dwellings, whose
    renters cannot be placed and are dropped.

    `mb_to_sa1` has MB_CODE_2021 and SA1_CODE_2021; `sa1_rented` is indexed
    by SA1_CODE_2021.
    """
    frame = mb_counts.merge(mb_to_sa1, on="MB_CODE_2021", how="left")
    sa1_dwellings = frame.groupby("SA1_CODE_2021")["Dwellings"].transform("sum")
    rented = frame["SA1_CODE_2021"].map(sa1_rented).fillna(0)
    share = (frame["Dwellings"] / sa1_dwellings).where(sa1_dwellings > 0, 0.0)
    frame[RENTED_ESTIMATED.count_type] = rented * share
    return frame.drop(columns="SA1_CODE_2021")


def carry_to_new_mesh_blocks(
    counts: pd.DataFrame,
    correspondence: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Move 2021 mesh block counts onto 2026 mesh blocks.

    `correspondence` has MB_CODE_2021, MB_CODE_2026 and RATIO_FROM_TO, the
    share of the 2021 mesh block falling in the 2026 one; ratios sum to 1 per
    2021 mesh block, so totals carry over. A 2026 mesh block with no 2021
    source gets no count.
    """
    merged = correspondence.merge(counts, on="MB_CODE_2021", how="inner")
    for column in columns:
        merged[column] = merged[column] * merged["RATIO_FROM_TO"]
    return merged.groupby("MB_CODE_2026", as_index=False)[columns].sum()


def read_correspondence(path: Path) -> pd.DataFrame:
    """Read an ABS mesh block correspondence, dropping its blank-source rows."""
    frame = pd.read_csv(path, dtype={"MB_CODE_2021": str, "MB_CODE_2026": str})
    frame = frame.dropna(subset=["MB_CODE_2021", "MB_CODE_2026"])
    frame["RATIO_FROM_TO"] = frame["RATIO_FROM_TO"].astype(float)
    return frame[["MB_CODE_2021", "MB_CODE_2026", "RATIO_FROM_TO"]]


def read_sa1_rented(path: Path) -> pd.Series:
    """2021 rented households per SA1, from the SA1 x LGA TableBuilder extract.

    The same extract `SA12021_LGA2021_C2021_Weightings.csv` is built from;
    summing over LGA gives each SA1's total.
    """
    extract = bw.read_universe([Path(path)], "Dwellings", "SA1", "LGA")
    return extract.groupby("SA1")["Dwellings"].sum().rename_axis(
        "SA1_CODE_2021"
    )


# ---------------------------------------------------------------- overlay


def state_labels(regions: pd.DataFrame, name: str, state: str) -> pd.Series:
    """Region labels, with a state suffix where a name repeats across states.

    Matches TableBuilder's 2011 LGA labels, so a file built here joins one
    built from TableBuilder: "Campbelltown (C)" in NSW and in SA become
    "Campbelltown (C) (NSW)" and "Campbelltown (C) (SA)".
    """
    names = regions[name].astype("string").str.strip()
    repeated = names.duplicated(keep=False)
    suffix = regions[state].astype(str).map(STATE_SUFFIXES)
    return names.where(~repeated, names + " (" + suffix + ")")


def assign_by_largest_overlap(
    mesh_blocks: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    states: gpd.GeoDataFrame,
    unincorporated: dict[str, str],
) -> pd.DataFrame:
    """Give each mesh block the region holding most of its area.

    `mesh_blocks` has MB_CODE; `regions` has CODE and STATE (one polygon per
    region; regions with no polygon are left out by the caller); `states` has
    STATE. `unincorporated` maps a state to the region code that takes the
    part of the state no region covers. All three frames must share an
    equal-area CRS.

    Returns one row per mesh block with a geometry: MB_CODE, CODE, SHARE (the
    winner's share of the mesh block's area), CANDIDATES (how many regions,
    including Unincorporated ones, hold any of it) and FALLBACK (True where
    no candidate held any area and the nearest region was used instead).
    """
    mesh_blocks = mesh_blocks[
        mesh_blocks.geometry.notna() & ~mesh_blocks.geometry.is_empty
    ].reset_index(drop=True)
    geoms = mesh_blocks.geometry.values
    mb_area = shapely.area(geoms)

    region_areas = _overlap_areas(geoms, regions.geometry.values)
    region_areas["CODE"] = regions["CODE"].to_numpy()[region_areas["right"]]
    region_areas["STATE"] = regions["STATE"].to_numpy()[region_areas["right"]]

    state_areas = _overlap_areas(geoms, states.geometry.values)
    state_areas["STATE"] = states["STATE"].to_numpy()[state_areas["right"]]
    covered = region_areas.groupby(["left", "STATE"])["area"].sum()
    state_areas = state_areas.set_index(["left", "STATE"])
    state_areas["area"] = (
        state_areas["area"]
        - covered.reindex(state_areas.index, fill_value=0.0)
    ).clip(lower=0.0)
    state_areas = state_areas.reset_index()
    state_areas["CODE"] = state_areas["STATE"].map(unincorporated)
    state_areas = state_areas.dropna(subset=["CODE"])

    # A sliver below a square metre is the two boundary sets disagreeing,
    # not a region holding part of the mesh block.
    options = pd.concat(
        [
            region_areas[["left", "CODE", "area"]],
            state_areas[["left", "CODE", "area"]],
        ],
        ignore_index=True,
    )
    options = options[options["area"] >= 1.0]

    candidates = options.groupby("left").size()
    best = (
        options.sort_values(["left", "area"], ascending=[True, False])
        .drop_duplicates("left")
        .set_index("left")
    )

    result = pd.DataFrame(
        {
            "MB_CODE": mesh_blocks["MB_CODE"].to_numpy(),
            "CODE": best["CODE"].reindex(range(len(mesh_blocks))).to_numpy(),
            "SHARE": (
                best["area"].reindex(range(len(mesh_blocks))).to_numpy()
                / mb_area
            ),
            "CANDIDATES": candidates.reindex(
                range(len(mesh_blocks)), fill_value=0
            ).to_numpy(),
        }
    )
    result["FALLBACK"] = result["CODE"].isna()
    if result["FALLBACK"].any():
        missing = np.flatnonzero(result["FALLBACK"].to_numpy())
        _, nearest = regions.sindex.nearest(geoms[missing], return_all=False)
        result.loc[missing, "CODE"] = regions["CODE"].to_numpy()[nearest]
        result.loc[missing, "SHARE"] = 0.0
    return result


def _overlap_areas(left, right) -> pd.DataFrame:
    """Intersection area of every intersecting (left, right) geometry pair.

    Most mesh blocks sit wholly inside one region, so pairs where the left
    geometry is covered by the right one take the left geometry's own area
    and skip the costly intersection.
    """
    tree = shapely.STRtree(right)
    left_idx, right_idx = tree.query(left, predicate="intersects")
    shapely.prepare(right)
    covered = shapely.covered_by(left[left_idx], right[right_idx])
    area = np.where(covered, shapely.area(left[left_idx]), 0.0)
    rest = np.flatnonzero(~covered)
    area[rest] = shapely.area(
        shapely.intersection(left[left_idx[rest]], right[right_idx[rest]])
    )
    return pd.DataFrame({"left": left_idx, "right": right_idx, "area": area})


# ---------------------------------------------------------------- build


def mesh_block_table(pair: reg.MeshBlockPair) -> tuple[pd.DataFrame, dict]:
    """One row per 2026 mesh block: its POA, its 2011 LGA and its counts.

    Returns the table and a dict of the diagnostics the README and the
    verification need (mesh blocks straddling a boundary, counts dropped).
    """
    counts = read_mesh_block_counts(pair.mb_counts)
    mb_to_sa1 = pd.read_excel(
        pair.mb_count_allocation,
        dtype=str,
        usecols=["MB_CODE_2021", "SA1_CODE_2021"],
    )
    sa1_rented = read_sa1_rented(pair.sa1_rented_extract)
    counts = estimate_rented_households(counts, mb_to_sa1, sa1_rented)
    count_columns = [u.count_type for u in MESH_BLOCK_UNIVERSES]
    carried = carry_to_new_mesh_blocks(
        counts, read_correspondence(pair.mb_correspondence), count_columns
    )

    poa = pd.read_excel(
        pair.geog1_allocation,
        dtype=str,
        usecols=["MB_CODE_2026", pair.geog1_allocation_column],
    )

    mesh_blocks = gpd.read_file(
        pair.mb_boundaries, columns=["MB_CODE26"]
    ).rename(columns={"MB_CODE26": "MB_CODE"})
    regions = gpd.read_file(pair.geog2_boundaries)
    regions["LABEL"] = state_labels(
        regions, pair.geog2_boundary_name, pair.geog2_boundary_state
    )
    regions = regions.rename(
        columns={
            pair.geog2_boundary_code: "CODE",
            pair.geog2_boundary_state: "STATE",
        }
    )
    states = gpd.read_file(pair.state_boundaries).rename(
        columns={pair.state_boundary_code: "STATE"}
    )
    unincorporated = {
        state: f"{state}9399" for state in states["STATE"].astype(str)
    }

    with_polygon = regions[regions.geometry.notna()]
    assigned = assign_by_largest_overlap(
        mesh_blocks.to_crs(EQUAL_AREA_CRS),
        with_polygon.to_crs(EQUAL_AREA_CRS),
        states[states.geometry.notna()].to_crs(EQUAL_AREA_CRS),
        unincorporated,
    )
    labels = regions.set_index("CODE")["LABEL"]
    assigned["LGA_LABEL"] = assigned["CODE"].map(labels)
    if assigned["LGA_LABEL"].isna().any():
        raise ValueError("a mesh block was given an LGA code with no label")

    table = (
        assigned.merge(
            poa, left_on="MB_CODE", right_on="MB_CODE_2026", how="left"
        )
        .merge(carried, on="MB_CODE_2026", how="left")
        .drop(columns="MB_CODE_2026")
    )
    table[count_columns] = table[count_columns].fillna(0)

    diagnostics = {
        "totals_2021": counts[count_columns].sum().to_dict(),
        "totals_carried": carried[count_columns].sum().to_dict(),
        "totals_placed": table[count_columns].sum().to_dict(),
        "fallback_mesh_blocks": int(table["FALLBACK"].sum()),
        "mesh_blocks": len(table),
    }
    for threshold in (0.999, 0.99, 0.9):
        straddling = table[table["SHARE"] < threshold]
        diagnostics[f"straddling_{threshold}"] = {
            "mesh_blocks": len(straddling),
            **straddling[count_columns].sum().to_dict(),
        }
    return table, diagnostics


def build_mesh_block_pair(
    pair: reg.MeshBlockPair,
    out_dir: Path = reg.WEIGHTINGS_DIR,
    write: bool = True,
    table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a mesh-block pair's weightings file, like build_pair does.

    Pass `table` (from mesh_block_table) to skip the slow read and overlay.
    """
    if table is None:
        table, _ = mesh_block_table(pair)
    combined = weightings_from_table(table, pair)
    if write:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        combined.to_csv(Path(out_dir) / pair.output_filename, index=False)
    return combined


def weightings_from_table(
    table: pd.DataFrame, pair: reg.MeshBlockPair
) -> pd.DataFrame:
    """Aggregate the mesh block table to geography pairs and weight it."""
    geog1 = pair.geog1_allocation_column
    table = table[
        table[geog1].str.fullmatch(r"\d{4}", na=False)
    ].copy()
    pairs = table.groupby([geog1, "LGA_LABEL"], as_index=False)[
        [u.count_type for u in MESH_BLOCK_UNIVERSES]
    ].sum()

    frames: dict[str, pd.DataFrame] = {}
    for universe in MESH_BLOCK_UNIVERSES:
        extract = pairs.loc[
            pairs[universe.count_type] > 0,
            [geog1, "LGA_LABEL", universe.count_type],
        ]
        frames[universe.key] = gw.calculate_geog_weighting(
            extract,
            geog1=geog1,
            geog1_name=pair.geog1_name,
            geog2="LGA_LABEL",
            geog2_name=pair.geog2_name,
            count_type=universe.count_type,
            count_name=universe.count_name,
            add_shapefile_variables=False,
        )

    combined = gw.combine_universes(
        frames, pair.geog1_name, pair.geog2_name, MESH_BLOCK_UNIVERSES
    )
    gw.check_weightings(
        combined, pair.geog1_name, pair.geog2_name, MESH_BLOCK_UNIVERSES
    )

    attributes = (
        table.drop_duplicates("LGA_LABEL")
        .set_index("LGA_LABEL")["CODE"]
        .rename(pair.geog2_boundary_code)
    )
    combined[pair.geog2_boundary_code] = combined[pair.geog2_name].map(
        attributes
    )
    return combined.sort_values(
        [pair.geog1_name, pair.geog2_name], ignore_index=True
    )
