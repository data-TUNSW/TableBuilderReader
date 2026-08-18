# TableBuilderReader read path: speed and footer correctness

Date: 2026-08-12

Fix two defects in `TableBuilderReader`'s CSV read path: it is forced onto the Python CSV
parser unnecessarily, and it fails to recognise the footer line used by current TableBuilder
exports, which leaks a junk row into every dataframe it produces.

Found while scoping a POA-to-SA1 weightings job. That job has since been dropped; this fix
stands on its own and affects every file the module reads.

## Problem 1: forced Python parser

`TableBuilderReader.__init__` passes `skipfooter` to `pd.read_csv`, which forces
`engine="python"`. That is a pure-Python parse of the entire file, roughly an order of
magnitude slower than the C parser and heavier on peak memory. It bites hardest on the large
cross-tabs this module exists to read: `2021_TBD_SA1_LGA_TEND-Rented.csv` is 607 MB, and the
POA cross-tabs are 20-30 MB and around 1M rows each.

`skipfooter` is not needed. `detect_footer_row` has already walked the whole file and knows
exactly where the data ends, so `nrows` can be computed instead and the default C parser used.

## Problem 2: unrecognised footer line

`expected_footer_strings` contains `"Data source: Census of Population and Housing"`, but
current TableBuilder exports end with
`"Dataset: Census of Population and Housing, 2021, TableBuilder"`. That line is not matched,
so `skipfooter` comes up one short and the line is parsed as a data row.

`clean_poa` then extracts `2021` from it via the `(\d{4})` regex, producing a phantom postcode
`2021` row with a NaN count. It reaches `self.df`, both entries in `self.df_changes`, and
`self.variables["POA"]["data"]`. It is only removed later, incidentally, because callers pass
`add_filter={"Dwellings": "> 0"}` and NaN compares false. Any caller that does not filter on
the count keeps the junk row.

Verified present in all three current cross-tabs:

| File | Total rows | `skipfooter` detected | First footer row | Rows read now | Rows read with `nrows` |
|---|---|---|---|---|---|
| `2021_TBD_POA_SA3_TEND-Rented.csv` | 922,427 | 7 | 922,419 | 922,408 | 922,407 |
| `2021_TBD_POA_SED(2025)_TEND-Rented.csv` | 1,168,226 | 7 | 1,168,218 | 1,168,207 | 1,168,206 |
| `2021_TBD_POA_LGA(2025)_TEND-Rented.csv` | 1,472,171 | 7 | 1,472,163 | 1,472,152 | 1,472,151 |

In each case the footer block is 8 rows but only 7 are detected, and the one extra row read
today is the `"Dataset: ..."` line.

## Change

1. Add `"Dataset: Census of Population and Housing"` to the default `expected_footer_strings`.
   With this the footer blocks above become contiguous 8-row blocks.

2. In `detect_footer_row`, additionally record the total row count and
   `self.data_end_row = min(self.footer_rows)`, then derive
   `self.nrows = self.data_end_row - self.variable_row - 1`. Data begins at
   `variable_row + 1` because `skiprows=variable_row` combined with `header=0` consumes the
   variable row itself.

3. Leave `self.skipfooter` populated as it is now. It is a documented attribute and external
   code may read it.

4. Guard against truncation. `nrows` stops at the *first* blank row, so a blank row inside the
   data would silently cut the table short where `skipfooter` would not. Check
   `total_rows - self.data_end_row == len(self.footer_rows)`. If it holds, use the fast path.
   If it does not, the footer block is not contiguous, so fall back to the existing
   `skipfooter` plus `engine="python"` path and emit a warning. The module then degrades to
   today's behaviour rather than losing rows.

5. If `self.footer_rows` is empty, set `self.nrows = None` and read to EOF.

6. Apply to both `pd.read_csv` calls, the `column_variable is None` branch and the
   `column_variable` branch, dropping `skipfooter=` and `engine="python"` from each on the
   fast path.

Not in scope: `detect_footer_row` still walks the whole file with `csv.reader`. At roughly
10-20 seconds for a million-row file this is not worth restructuring, and scanning backward
from EOF would be a larger change than this justifies.

## Verification

Regression against raw files already on disk, exercising each read path:

| File | Path exercised |
|---|---|
| `2021_TBD_POA_SA3_TEND-Rented.csv` | two-geography long |
| `2021_TBD_POA_SED(2025)_TEND-Rented.csv` | two-geography long, bracketed variable |
| `2021_TBD_POA_LGA(2025)_TEND-Rented.csv` | two-geography long |
| `2021_TBP_SA1_TEND-Rented.csv` | single geography, different preamble shape |

The criterion is **not** plain equality. Expect exactly one difference per affected file: the
phantom postcode `2021` / NaN-count row present under the old code and absent under the new.
Each difference must be enumerated and confirmed to be that row and nothing else. Compare
`df_changes["Original"]` and `df_changes["Clean"]`, not just the final `df`, since the `> 0`
filter masks the bug in the final frame. Also check `self.variables["POA"]["data"]` no longer
carries the phantom value.

Record before/after wall-clock on the largest file used, so the speedup is measured rather
than assumed.

The `column_variable` wide-to-long `stack` branch has **no caller anywhere in this repo**;
`grep` finds no usage in either notebook. It gets the same `nrows` substitution but cannot be
regression-tested against a known-good output. This is a residual risk, must be stated in the
commit message, and must not be described as verified. If a wide-format export turns up later
it needs retesting.

## Delivery

Own branch, own commit. `table_builder_reader.py` is public and submoduled into other
projects, so it needs to be reviewable and revertable on its own.

Downstream note for the commit message: any consumer that relied on the phantom row being
filtered out by a count-based `add_filter` is unaffected, but row counts in unfiltered reads
will drop by one per file.
