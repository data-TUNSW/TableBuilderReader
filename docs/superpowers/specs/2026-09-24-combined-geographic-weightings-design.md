# Combined Geographic Weightings

**Date:** 2026-09-24
**Status:** Design approved, not yet implemented

## Purpose

Replace the one-file-per-universe geographic weightings in
`G:/Shared drives/Data/ABS/Geography/processed/` with one file per geography pair
carrying every available weighting as columns.

Today a project that needs both household and person weightings for POA to LGA
opens two files, joins them, and reconciles two row sets that do not match. There
are four such files for POA to SED and none for some pairs, so which universes
exist for which pair is only discoverable by listing the directory. The combined
format makes a pair one file and one read, and makes the available universes
visible in its header.

The existing `processed/` files are not touched, moved or deleted. Consuming
projects migrate one at a time, when someone next works on them.

## Background

Every current weightings file is produced by `calculate_geog_weighting` in
`src/geographic_weightings.ipynb` and shares a seven-column core:

```
<geog1 name>, <geog2 name>, <count>,
<geog1 name> Total <count>, <geog2 name> Total <count>,
<geog1 name> Fraction, <geog2 name> Fraction,
[+ shapefile attribute columns]
```

The only thing that varies between the files of a pair is the universe: the count
column is `Rented Households`, `Renters (People)`, `Households` or `Persons`. Each
universe is a separate TableBuilder extract read with an `add_filter` of `> 0`, so
their row sets differ.

The two fraction columns are the reason these files exist, and the reason they are
easy to misuse. `<geog1> Fraction` is the share of geog1 falling in geog2;
`<geog2> Fraction` is the share of geog2 made up by geog1. Picking the wrong one
produces plausible numbers that are wrong.

## The universe grid

Six universes. Each is a separate TableBuilder extract, read with `> 0` on its
count column.

| Universe | Count column | TableBuilder filter | Basis |
|---|---|---|---|
| All Households | `Households` | `DWTD = Occupied` | Dwellings (TBD) |
| All Persons | `Persons` | none | Persons (TBP) |
| Rented Households | `Rented Households` | `TEND = Rented` | Dwellings (TBD) |
| Renters (People) | `Renters (People)` | `TEND = Rented` | Persons (TBP) |
| Private Rented Households | `Private Rented Households` | `TENLLD = Rented: Real estate agent` | Dwellings (TBD) |
| Private Renters (People) | `Private Renters (People)` | `TENLLD = Rented: Real estate agent` | Persons (TBP) |

All counts are on a **place of enumeration** basis. Households are occupied
private dwellings, a Census Night concept: a dwelling whose usual residents were
all away is classified unoccupied and excluded. There is no usual-residence
equivalent, so persons are kept on the same basis for consistency.

Counts are randomly perturbed by the ABS, so cross-tab margins will not exactly
match published totals for either geography. Fractions are computed from the
cross-tab's own margins, so they still sum to 1 within each geography.

### Current coverage

Files are built from the extracts that exist. Universes without an extract are
absent from the file and are added by pulling the extract and re-running. Coverage
as at 2026-09-24:

| Pair | All HH | All P | Rented HH | Rented P | Priv HH | Priv P |
|---|---|---|---|---|---|---|
| POA to SED (2025) | yes | yes | yes | yes | no | no |
| POA to CED (2025) | yes | yes | no | no | no | no |
| POA to LGA (2025) | no | no | yes | yes | no | no |
| POA to SA3 | no | no | yes | yes | no | no |
| POA to SAL | no | no | no | no | no | no |
| SA1 to LGA | no | no | yes | no | no | no |

No POA-by-geog2 TENLLD cross-tab exists yet, so both private renter universes
start empty across every pair.

## File layout

A new top-level directory, a sibling of `processed/` rather than inside it, so
that a project globbing `processed/*.csv` cannot pick up a combined file by
accident and the canonical path is unambiguous:

```
G:/Shared drives/Data/ABS/Geography/weightings/
  POA_LGA(2025)_Weightings.csv
  POA_SED(2025)_Weightings.csv
  POA_CED(2025)_Weightings.csv
  POA_SA3_Weightings.csv
  POA_SAL_Weightings.csv
  SA1_LGA_Weightings.csv
  README.md
```

Naming is `{GEOG1}_{GEOG2}[(vintage)]_Weightings.csv`. Vintage stays in the
filename rather than becoming a column, matching the existing convention; a
boundary revision produces a new file, not new rows.

`README.md` is written by the generator, not by hand. It carries the format spec,
the coverage grid above regenerated from what was actually built, and the
generation date for each file.

## Column specification

Identity columns, then one five-column block per available universe, then
shapefile attributes once:

```
Postcode
Local Government Area (2025)

Rented Households
Postcode Total Rented Households
Local Government Area (2025) Total Rented Households
Postcode Fraction Rented Households
Local Government Area (2025) Fraction Rented Households

Renters (People)
Postcode Total Renters (People)
Local Government Area (2025) Total Renters (People)
Postcode Fraction Renters (People)
Local Government Area (2025) Fraction Renters (People)

POA_CODE21
POA_NAME21_AREASQKM21
LGA_CODE25
STE_CODE21
STE_NAME21
LGA_NAME25_AREASQKM
```

Count and total column names are unchanged from the existing files. Only the
fraction columns gain the universe suffix, because they are the only names that
collide across universes: `Postcode Total Rented Households` and
`Postcode Total Persons` are already distinct, but both files call their fraction
`Postcode Fraction`. Migrating a project is therefore a path change plus renaming
two columns per universe.

Universe blocks appear in the fixed order of the grid table above, so a file's
column order is predictable from the set of universes it has. Universes without
an extract are absent entirely; no all-null placeholder block is written.

Two shapefile columns are dropped as exact duplicates of identity columns:
`POA_NAME21` (identical to `Postcode`) and `LGA_NAME25` (identical to the LGA
column). Codes, area columns and state columns are kept. The equivalent applies
to other pairs: drop the geog2 shapefile name column, keep its code.

A pair may have no shapefile on one or both sides, in which case that side
contributes no attribute columns. `SA1_LGA_Rented_Weightings.csv` has none today,
though `SA1_2021_AUST_GDA2020.shp` exists and can be wired into the registry.

## Joining the universes

Outer join on the `(geog1, geog2)` pair. Row sets differ between universes, since
each extract is filtered to `> 0` on its own count: a postcode-LGA sliver can
have occupied dwellings but no renters, so it appears in the All Households
extract and not the Rented one.

For a pair missing from a universe:

- **Count** is filled with `0`. The pair genuinely contributes nothing to that
  universe.
- **Totals** are filled from the totals table for that geography, not from the
  joined row, so they are never null while the geography exists in the universe
  at all.
- **Fractions** are filled with `0` where the corresponding total is above zero,
  and left **NaN** where that geography has no presence in the universe at all
  (the 0/0 case).

That NaN is deliberate. It distinguishes "this postcode contributes 0% of the
LGA's renters" from "this postcode has no renters to contribute". A consumer
reallocating data must check the total before trusting a fraction, and a NaN makes
failing to do so loud instead of silent.

## Chunked extracts

Some cross-tabs exceed TableBuilder's 40,000,000 cell limit. POA by SAL is
2,643 x 15,343 = 40,551,549 cells, over by 1.4%.

A universe's source in the registry is therefore **a list** of extract paths, not
a single path. `build_pair` concatenates the chunks into one national frame and
only then computes totals and fractions. The code is agnostic about how the
extract was chunked.

Two rules govern chunking, and both are correctness issues rather than
conveniences:

**Totals are computed after concatenation, never per chunk.** Some postcodes cross
state borders: 0872 spans NT, SA and WA, and 2611 and 2620 straddle NSW and the
ACT. If `Postcode Fraction` were computed within a per-state chunk, a cross-border
postcode's fractions would sum to 1 within each chunk it appears in, and to more
than 1 nationally. The counts would look correct and the weights would be quietly
wrong.

**Only the large side is chunked.** POA stays unrestricted in every extract;
only SAL is state-restricted. A Victorian suburb crossed against all national
postcodes still captures a NSW postcode reaching over the border. Restricting both
sides to the same state would drop those pairs entirely.

### POA to SAL extract naming

Per state, following the existing extract naming convention, with the state
attached to `SAL` as a filter value so that one glob selects exactly one
universe's chunks - for five of the six universes; All Persons is the
exception, and why is covered below:

| Universe | Filename pattern |
|---|---|
| All Households | `2021_TBD_POA_SAL-{STE}_DWTD-Occupied.csv` |
| All Persons | `2021_TBP_POA_SAL-{STE}.csv` |
| Rented Households | `2021_TBD_POA_SAL-{STE}_TEND-Rented.csv` |
| Renters (People) | `2021_TBP_POA_SAL-{STE}_TEND-Rented.csv` |
| Private Rented Households | `2021_TBD_POA_SAL-{STE}_TENLLD-Rented Real estate agent.csv` |
| Private Renters (People) | `2021_TBP_POA_SAL-{STE}_TENLLD-Rented Real estate agent.csv` |

`{STE}` is one of `NSW`, `VIC`, `QLD`, `SA`, `WA`, `TAS`, `NT`, `ACT`, `OT`.
Nine chunks per universe, 54 extracts for the full grid. Suburb counts per state
are NSW 4,543, Vic 2,945, Qld 3,234, SA 1,697, WA 1,700, Tas 777, NT 304,
ACT 137, Other Territories 6; the largest chunk is 12.0M cells, 30% of the limit.

The registry stores a glob per universe for five of the six universes, so
adding or re-pulling a state for those needs no code change. All Persons is
deliberately exempt: its filename (`2021_TBP_POA_SAL-{STE}.csv`) has nothing
after the state code to disambiguate it from the other person-basis
universes' per-tenure chunks, so a `{STE}-*` glob would also match
`2021_TBP_POA_SAL-NSW_TEND-Rented.csv` and the `TENLLD-...` chunk, silently
inflating All Persons' totals without tripping the fractions-sum-to-1 check.
`_poa_sources` instead builds its source list explicitly from `CHUNK_STATES`,
so a new or renamed state needs a one-line edit there.

## Generator

`calculate_geog_weighting` moves out of the notebook into
`src/abs_methods/geographic_weightings.py` as module code. The notebook is left
untouched as the record of how the `processed/` files were produced.

The module holds:

- **A registry** of pairs. Each entry gives geog1 and geog2 codes and display
  names, the vintage, both shapefile paths and name columns, and a mapping from
  universe to source extract glob. Absent universes are simply absent from the
  mapping.
- **`calculate_geog_weighting(...)`** — unchanged behaviour, moved verbatim. It
  still produces a single-universe frame.
- **`build_pair(pair)`** — for each universe in the registry entry, resolves the
  glob, concatenates the chunks, calls `calculate_geog_weighting`, then
  outer-joins the per-universe frames on the identity columns, applies the fill
  rules, orders the columns, and runs the QA checks.
- **`build_all()`** — writes every pair plus `README.md`.

Adding a universe once its extract exists is one registry line and a re-run.

### QA checks

Run on every write, raising rather than warning:

1. Each `<geog1> Fraction <universe>` sums to 1.0 within each geog1 whose total is
   above zero, within floating point tolerance. Same for geog2. This is the check
   that catches a mis-chunked extract: a botched concatenation shows up as a
   postcode summing to 2 or 3, rather than as plausible-looking bad numbers.
2. No duplicate `(geog1, geog2)` rows.
3. Every count column is non-negative, and each count is no greater than either of
   its two totals.
4. Every fraction is in [0, 1] or NaN.

## Skill

`~/.claude/skills/abs-geographic-weightings/SKILL.md`, at user level so it is
available across every project on this machine.

Triggers on reallocating or converting data between ABS geographies, on postcode
to or from LGA, SED, CED, SA3, SA1 and suburb, and on the words weighting,
concordance and correspondence.

Contents:

- The canonical path, and that `processed/` is legacy and being retired.
- The universe list and how to choose: match the universe to what is being
  reallocated, so rent data by a renter universe and population counts by persons.
- The current coverage grid, or a pointer to the generated README for it.
- The direction rule, which is the thing most likely to be got wrong:

  > To push **postcode** data **up** to LGA, weight by
  > `Postcode Fraction <universe>` and sum by LGA. It is the share of that
  > postcode falling in that LGA.
  >
  > To push **LGA** data **down** to postcodes, weight by
  > `LGA Fraction <universe>` and sum by postcode.

- The NaN convention and why checking the total matters.
- How to regenerate, and the chunking rules for extracts over the cell limit.
- Which projects still read the legacy files, so a migration can be recognised
  as such when one is encountered.

A block in `~/.claude/CLAUDE.md` points at the canonical path and the skill, so
that a Claude working in any project knows the file exists before it reaches for
`processed/`.

## Migration

Nothing in the consuming projects is changed by this work. Four read the legacy
files today:

- **Bond Loss Risk Factor Analysis** — `src/build_model_data.py`,
  `src/build_model_data/build_model_data.py`, `src/build_model_data/sync_data.py`,
  `notebooks/notebook.ipynb`, and three files under `docs/`. The heaviest user.
- **Rental Vacancy** — three `.Rmd` files under `analysis/`. The R side needs the
  skill to read naturally, not just the Python side.
- **Rental Affordability** — docs and TODO only; not yet wired up, so it should be
  pointed at the new files from the start.
- **Archive/Geography** — dead, ignore.

Each migrates when someone next works on it. A migration is: change the path,
rename the two fraction columns to their universe-suffixed names, and drop the
join against a second weightings file if the project was using two.

## Out of scope

- Changing, moving or deleting anything in `processed/`.
- Editing the consuming projects.
- Pulling the missing TableBuilder extracts. That is manual work in the
  TableBuilder UI, done alongside, and the files appear as the extracts land.
- Older vintages (LGA 2016, SED 2016/2022 and similar). The registry can hold them
  later; the initial build covers the pairs listed above.
- `geometry` columns. `add_geometry` stays False; consumers needing geometry read
  the shapefile.

## Open items

None blocking. The private renter universes and the POA to SAL extracts are
expected to arrive after the first build, which the design already accommodates.
