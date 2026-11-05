# Handover: TableBuilder extracts for postcode weightings

Written 2026-11-05 for a Claude session driving TableBuilder in Chrome (`/chrome`).
Your job is to **download and file extracts**. Turning them into weightings files
(registering pairs in `src/abs_methods/weightings_registry.py`) is a separate step
in this repo, done afterwards; do not edit code here.

## Goal

Postcode (POA) by each major geography, for every Census in TableBuilder, in
all six count universes. Each extract becomes one universe of one pair in
`G:\Shared drives\Data\ABS\Geography\weightings\`, named
`POA<edition>_<GEOG><edition>_C<census>_Weightings.csv`.

Target geographies: **LGA, SA4, SA3, GCCSA, SED, CED.**

## Ground rules

- **The user logs in to TableBuilder.** Never type, read or store credentials.
  If the session times out, stop and ask the user to log in again.
- **Never overwrite a file.** If the target name already exists in the raw
  folder, stop and compare; existing files are already in use.
- Work in the priority order below and file each extract as you go, so a stop
  partway through leaves finished, usable files.
- Before you build the first table, open the reference files below and match
  their header exactly: dataset title, "Counting" line, Filters block, the
  `(EN)` geography labels and the layout.

## Table recipe (every extract)

| Setting | Value |
|---|---|
| Dataset | Census of Population and Housing, `<year>`, **place of enumeration**. Dwellings dataset for `TBD` files, persons dataset for `TBP` files |
| Rows | POA, then the target geography (both in rows, POA outermost) |
| Columns / wafers | none |
| Filter | per universe, below; none for the all-persons universe |
| Output | CSV, TableBuilder's default layout, totals left as TableBuilder gives them |

Reference files to match (2021, in `G:\Shared drives\Data\ABS\2021 Census\raw\`):

- `2021_TBD_POA_LGA(2025)_TEND-Rented.csv`: dwellings, filtered
- `2021_TBP_POA_SED(2025).csv`: persons, no filter
- `2021_TBD_POA_SED(2025)_DWTD-Occupied.csv`: dwellings, occupied filter
- `2021_TBP_POA_TENLLD-Rented Real estate agent.csv`: the private-rental filter label

### The six universes

| # | File type | Filter (2021 variable and category) | Filename suffix |
|---|---|---|---|
| 1 | `TBD` | DWTD Dwelling Type = Occupied private dwellings | `_DWTD-Occupied` |
| 2 | `TBP` | none | (nothing) |
| 3 | `TBD` | TEND Tenure Type = Rented | `_TEND-Rented` |
| 4 | `TBP` | TEND Tenure Type = Rented | `_TEND-Rented` |
| 5 | `TBD` | TENLLD Tenure and Landlord Type = Rented: Real estate agent | `_TENLLD-Rented Real estate agent` |
| 6 | `TBP` | TENLLD Tenure and Landlord Type = Rented: Real estate agent | `_TENLLD-Rented Real estate agent` |

Older Censuses name these variables differently (2011 has `LLDD Landlord Type`
with category "Real estate agent", and counts "Dwellings, Location on Census
Night"). Pick the variable that gives the same population, keep the filename
suffix above unchanged so the generator's patterns match, and record the actual
variable and category in the manifest.

## Filenames and where they go

```
<census>_TB<D|P>_POA_<GEOG>[(<edition>)]<suffix>.csv
```

Saved to `G:\Shared drives\Data\ABS\<census> Census\raw\`.

- LGA, SED and CED carry their boundary edition in brackets, because a Census
  dataset can offer several: `2021_TBD_POA_LGA(2025)_TEND-Rented.csv`.
- SA4, SA3 and GCCSA carry none; they only come in the Census year's ASGS:
  `2021_TBP_POA_SA4_TENLLD-Rented Real estate agent.csv`.
- Downloads land in the user's Downloads folder; move and rename each one, then
  check its first ten lines match what you asked for.

## What to pull, in priority order

Before starting each Census, list which LGA, SED and CED editions its dataset
offers and record them in the manifest.

### 1. 2021, SA4 and GCCSA (critical path for the Rental Affordability project)

All six universes for each: **12 files.**

### 2. 2021, complete the existing pairs

Already on disk (do not re-pull):

| Pair | Have |
|---|---|
| LGA(2025) | 3 TEND-Rented (TBD), 4 TEND-Rented (TBP) |
| SED(2025) | 1, 2, 3, 4 |
| CED(2025) | 1, 2 |
| SA3 | 3, 4 |

Missing: LGA(2025) 1, 2, 5, 6; SED(2025) 5, 6; CED(2025) 3, 4, 5, 6;
SA3 1, 2, 5, 6. **14 files.**

### 3. 2021, newer editions (only if offered)

If the 2021 dataset now offers LGA, SED or CED editions newer than 2025 (ASGS
Edition 4 released 2026 boundaries from July 2026), pull all six universes for
each newer edition: up to **18 files.** Skip and note it in the manifest if
none are offered.

### 4. 2016

POA 2016 by LGA, SA4, SA3, GCCSA, SED, CED, using the newest edition the 2016
dataset offers for LGA, SED and CED. Six universes each: **36 files.**

### 5. 2011

The same as 2016 for the 2011 dataset: **36 files.**

### 6. 2006 (lowest priority; ask the user before starting)

2006 uses the older ASGC: no SA4, SA3 or GCCSA exist. If the user wants it,
pull POA by LGA, SED and CED only, six universes each: up to **18 files.**

2026 Census data is not out yet (first release mid-2027). Nothing to pull for it.

## Limits

TableBuilder caps a table at 40,000,000 cells. Every table here is about 2,600
postcodes by under 600 areas, far under it. If one is refused for size, stop and
report rather than chunking; only POA by SAL needs chunking, and that is out of
scope.

## Manifest

Append one row per filed extract to
`G:\Shared drives\Data\ABS\Geography\weightings_extracts_manifest.csv`
(create it with this header if absent):

```
filename,census,geog,edition,universe,dataset_title,filter_variable,filter_category,downloaded,notes
```

Record skipped items too (with a note), so the next session knows what was
checked and not offered.

## Handing back

Report: files filed per Census, anything skipped and why, the editions each
dataset offered, and any variable-name differences in older Censuses. The
follow-up in this repo is registering pairs for what landed, and checking that
the reader handles the older Censuses' header wording (2016 and 2011 say
"Dwellings, Location on Census Night", not "Dwelling Records").
