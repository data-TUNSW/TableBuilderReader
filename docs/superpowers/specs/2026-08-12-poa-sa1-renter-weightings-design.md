# POA to SA1 Private Renter Weightings

Date: 2026-08-12

Produce postcode-to-SA1 private renter weightings for NSW, matching the format of the
existing `POA_SA3` / `POA_LGA(2025)` / `POA_SED(2025)` weightings files. Two parts: a
performance and correctness fix to `TableBuilderReader`, which is a prerequisite, and the
weightings themselves.

## Background

`src/rented_weightings.ipynb` builds geography weightings by reading a two-geography ABS
TableBuilder cross-tab through `TableBuilderReader`, then passing the cleaned dataframe to
`calculate_geog_weighting`, which computes each geography's total and each row's fraction of
those totals, joins shapefile attributes for both geographies, and writes a CSV to
`G:/Shared drives/Data/ABS/Geography/processed/`.

Two facts about SA1 shape this work.

**SA1s nest inside postcodes.** ASGS non-ABS structures, POA among them, are built by
allocating whole SA1s. The existing `SA1_LGA_Rented_Weightings.csv` confirms this
empirically: 59,060 of 59,239 rows have an `SA1 Fraction` of exactly 1.0, with 89 SA1s
splitting across LGAs, consistent with TableBuilder perturbation rather than real geography.
A POA-to-SA1 file will therefore carry `SA1 Fraction = 1.0` on effectively every row. Only
`Postcode Fraction` is informative. The file is a concordance with a weight attached, and it
is being produced in the standard two-fraction format for consistency with its siblings, not
because both fractions carry information.

**The cross-tab is dense.** TableBuilder emits the full cross-product. Australia-wide this is
61,845 SA1s by 2,644 POAs, or 163.5M cells, which extrapolates from the 33.8M-cell / 607 MB
`2021_TBD_SA1_LGA_TEND-Rented.csv` to roughly 3 GB and would most likely be refused outright.
Restricting to NSW brings it to about 12.8M cells and roughly 230 MB, comparable to work
already done.

## Scope decisions

- **Coverage: NSW.** Not Australia-wide, for the size reason above.
- **Tenure: `TENLLD` = `Rented: Real estate agent`**, i.e. private renters, not the `TEND` =
  `Rented` used by the sibling files.
- **Counts: both.** Dwellings (`TBD`) and persons (`TBP`), matching every other POA
  weightings pair.
- **Primary use: distributing postcode-level data down to SA1s**, which consumes
  `Postcode Fraction`.

## Part A: `TableBuilderReader` read path

### Problem

`TableBuilderReader.__init__` passes `skipfooter` to `pd.read_csv`, which forces
`engine="python"`. On a 12.8M-row file this is a pure-Python parse taking on the order of
twenty minutes, with high peak memory. `detect_footer_row` has already walked the file and
knows exactly where the data ends, so `nrows` can be computed instead and the C parser used.

There is also a latent correctness bug. `expected_footer_strings` contains
`"Data source: Census of Population and Housing"`, but the 2025-vintage exports end with
`"Dataset: Census of Population and Housing, 2021, TableBuilder"`. That line is not
recognised as footer, so `skipfooter` is one short and the line is parsed as a data row.
`clean_poa` then extracts `2021` from it, producing a phantom postcode `2021` row with a NaN
count. It reaches `self.df`, `self.df_changes` and `self.variables["POA"]["data"]`, and is
only removed later, incidentally, because `add_filter={"Dwellings": "> 0"}` evaluates false
against NaN. Verified present in all three of `2021_TBD_POA_SA3_TEND-Rented.csv`,
`2021_TBD_POA_SED(2025)_TEND-Rented.csv` and `2021_TBD_POA_LGA(2025)_TEND-Rented.csv`.

### Change

1. Add `"Dataset: Census of Population and Housing"` to the default
   `expected_footer_strings`. With this, the footer block in the three files above becomes
   contiguous at 8 rows rather than a detected 7 with a gap.

2. In `detect_footer_row`, additionally record the total row count and
   `self.data_end_row = min(self.footer_rows)`, then derive
   `self.nrows = self.data_end_row - self.variable_row - 1`. Data begins at
   `variable_row + 1` because `skiprows=variable_row` combined with `header=0` consumes the
   variable row itself. Leave `self.skipfooter` populated as now; it is a documented
   attribute and other code may read it.

3. Guard against truncation. `nrows` stops at the *first* blank row, so a blank row appearing
   inside the data would silently cut the table short, where `skipfooter` would not. Assert
   the footer block is contiguous by checking
   `total_rows - self.data_end_row == len(self.footer_rows)`. If it holds, use the fast path.
   If it does not, fall back to today's `skipfooter` plus `engine="python"` path and emit a
   warning, so the module degrades to current behaviour rather than losing rows.

4. If `self.footer_rows` is empty, set `self.nrows = None` and read to EOF.

5. Apply to both `pd.read_csv` calls, the `column_variable is None` branch and the
   `column_variable` branch, dropping `skipfooter=` and `engine="python"` from each on the
   fast path.

Not in scope: `detect_footer_row` still walks the whole file with `csv.reader`. At around
10-20 seconds for 12.8M rows this is not worth restructuring, and scanning backward from EOF
would be a larger change than this work justifies.

### Verification

Regression against the raw files already on disk, exercising each read path:

| File | Path exercised |
|---|---|
| `2021_TBD_POA_SA3_TEND-Rented.csv` | two-geography long |
| `2021_TBD_POA_SED(2025)_TEND-Rented.csv` | two-geography long, bracketed variable |
| `2021_TBD_POA_LGA(2025)_TEND-Rented.csv` | two-geography long |
| `2021_TBP_SA1_TEND-Rented.csv` | single geography, different preamble shape |

The `column_variable` wide-to-long `stack` branch has **no caller anywhere in this repo**;
`grep` finds no usage in either notebook. It therefore gets the same `nrows` substitution but
cannot be regression-tested against a known-good output. This is a residual risk and should
be stated as such in the commit message. If a wide-format export turns up later, it needs
retesting. Do not claim that branch is verified.

The criterion is **not** plain equality. Expect exactly one difference per affected file: the
phantom postcode `2021` / NaN-count row is present under the old code and absent under the
new. Each difference must be enumerated and confirmed to be that row and nothing else.
Compare `df_changes["Original"]` and `df_changes["Clean"]`, not just the final `df`, since
the `> 0` filter masks the bug in the final frame.

Also record before/after wall-clock on the largest available file to confirm the speedup is
real rather than assumed.

### Delivery

Own branch, own commit, separate from any notebook work. `table_builder_reader.py` is public
and submoduled into other projects, so it needs to be reviewable and revertable on its own.

## Part B: POA to SA1 weightings

### Downloads

Two TableBuilder extracts, saved to `G:/Shared drives/Data/ABS/2021 Census/raw/`:

```
2021_TBD_POA_SA1(NSW)_TENLLD-Rented Real estate agent.csv
2021_TBP_POA_SA1(NSW)_TENLLD-Rented Real estate agent.csv
```

Table definition:

- Counting: Dwelling Records (`TBD`) and Person Records (`TBP`) respectively.
- Rows: `POA (EN)` first, then `SA1 (EN)` nested underneath it. Both in Row, nothing in
  Column. This yields the `"POA (EN)","SA1 (EN)",` long layout the reader handles; a Column
  layout would need the `column_variable` path and 19,749 columns is likely over
  TableBuilder's limit.
- Filter: `TENLLD Tenure and Landlord Type` = `Rented: Real estate agent`.
- POA selection: NSW POAs (609), ACT POAs (23), Other Territories POAs (3) and Cross Border
  POAs (15), totalling 650.
- SA1 selection: NSW only, 19,749.

The size reduction must come from the **geography selections**, not from a state filter.
Selecting Australia and filtering to NSW still enumerates the full 163.5M-cell cross-product
padded with zeros.

Including ACT, Other Territories and Cross Border POAs is deliberate. Because the SA1
selection is NSW-only these mostly return zeros, but they ensure no NSW SA1 is orphaned out
of a postcode that straddles a border.

### Filename parsing

Traced through the reader. `filtered_variables` splits on `_`, takes tokens containing `-`,
and yields `{TENLLD: "Rented Real estate agent"}`. `set_variables` splits after `TBD_` into
`['POA', 'SA1(NSW)', 'TENLLD-Rented Real estate agent']`, resolves the bracket to variable
name `SA1` with description `SA1 NSW` by the same path as the existing `SED(2025)` files, and
drops the token matching the `TENLLD` filter key. Net variables: `['POA', 'SA1']`.

`TENLLD-Rented Real estate agent` matches the convention already used by eleven files in
`raw/`. A `_STATE-NSW` suffix is deliberately **not** used: it would parse cleanly but is
factually wrong, since ACT and Other Territories POAs are in the selection. The `(NSW)`
bracket marks the SA1 scope, and the POA scope follows from it.

### `calculate_geog_weighting` changes

Three changes in `src/rented_weightings.ipynb`.

**1. Configurable count rename.** The function currently hardcodes `Dwellings` to
`Rented Households` and `Persons` to `Renters (People)`. Under a `TENLLD` filter those labels
overstate the data, which covers private rentals through an agent rather than all rentals.
Add a `count_rename` argument defaulting to the current mapping so existing cells are
unaffected, and pass for this run:

- `Dwellings` to `Private Rented Households`
- `Persons` to `Private Renters (People)`

**2. SA1 join dtype.** SA1 is the only geography joined on a code rather than a name.
TableBuilder writes it as a bare 11-digit number so pandas reads `int64`, while `SA1_CODE21`
in the shapefile is a string. Left alone the merge silently returns all-NaN shapefile
columns. Before each shapefile merge, align dtypes only when they differ, casting the
weighting column to the shapefile column's dtype. Route any numeric source through
`astype("int64").astype(str)` so a float column cannot produce `"10102100701.0"`. SA1 codes
begin with the state digit and carry no leading zeros, so the round-trip is exact. POA is
already a string by then via `clean_poa`, which matters because POA codes such as `0800` do
have leading zeros.

**3. SA1 has no name column.** Pass `geog2_shapefile_name="SA1_CODE21"`. The existing cleanup
drops `AUS_*` and `LOCI_URI21` and renames `AREASQKM21` to `SA1_CODE21_AREASQKM21`. SA2, SA3,
SA4, GCCSA and STE codes and names come along for free.

### Notebook cells

A new `## POA -> SA1 (NSW)` section following the existing pattern: a
`### Private Rented Household Weightings` cell and a `### Private Renter Weightings` cell,
each constructing a `TableBuilderReader` with `add_filter={"Dwellings": "> 0"}` /
`{"Persons": "> 0"}`, calling `calculate_geog_weighting` with `geog1="POA"`,
`geog1_name="Postcode"`, `geog2="SA1"`, `geog2_name="SA1"`, and writing to CSV behind the
`save` flag used by the SA3 cells.

### Outputs

Written to `G:/Shared drives/Data/ABS/Geography/processed/`:

```
POA_SA1(NSW)_Private_Rented_Household_Weightings.csv
POA_SA1(NSW)_Private_Renter_Weightings.csv
```

Columns: `Postcode`, `SA1`, the count, `Postcode Total <count>`, `SA1 Total <count>`,
`Postcode Fraction`, `SA1 Fraction`, then POA and SA1 shapefile attributes.

### Known characteristics of the result

These are properties of the data, not defects to fix.

- `Postcode Fraction` is the usable column. It gives each SA1's share of its postcode's
  private renters and sums to 1 within a postcode.
- `SA1 Fraction` will be 1.0 on essentially every row, for the nesting reason above. It
  exists for format consistency.
- The `> 0` filter drops SA1s with no private rentals, so they will not appear at all.
  Anything distributed downward implicitly allocates them nothing. This matches the sibling
  files' behaviour.
- Counts are randomly perturbed by the ABS. Small cells should not be relied on individually,
  and perturbation is the likely explanation for any SA1 that does appear against more than
  one postcode.

## Sequencing

Part A does not depend on the downloads and is a prerequisite for Part B, so it runs while
TableBuilder builds the extracts. Part B follows once the files land.
