# TODO

<!-- Live tracker, NOT an append-only log: items physically MOVE between the four
     sections below as their status changes.

     Sections, fixed order:
       ## Working on    ## To do    ## Later / Someday    ## Done (last 60 days)
     Lifecycle:  To do / Later  ->  Working on  ->  Done

     Each item: a `###` imperative title, a one-line task description (task only;
     no findings, results, or method notes), then ONE italic stamp line. Stamp
     fields are labelled, `·`-separated, one line, absolute YYYY-MM-DD dates; the
     final field is the kebab-case slug, which is also the branch name. Templates:
       To do / Later:  _added 2026-07-01_
       Working on:     _added 2026-07-01 · started 2026-07-01 · branch `slug` · slug_
       Done:           _added 2026-07-01 · started 2026-07-01 · done 2026-07-05 · branch `slug` · slug_ -->

## Working on

### Add POA-to-GCCSA weightings

- Register a POA x GCCSA pair in `weightings_registry.py` and generate the combined
  weightings file for it (Census 2021; 2016 too if the registry supports it), so
  postcode data can be reallocated to Greater Sydney and the rest of each state. The
  `*_POA_GCCSA_*` TableBuilder extracts for both Censuses are already in the raw folders.
  Requested by the Bond Loss cluster demographics work, which needs a Greater Sydney
  comparator.

_added 2026-11-06 · started 2026-11-07 · branch `poa-gccsa-weightings` · poa-gccsa-weightings_

## To do

### Find why national totals differ from a direct parse on a large multi-variable table

- For `2021_TBP_POA_RPIP_HCFMD(2 digit)_BPLP(2 digit)_TENLLD-REA.csv` (12.6m cells, three
  stacked variables, BPLP in columns), the reader's national total of primary-family
  reference persons was about half a direct parse of the same file (1.18m against 2.39m),
  while totals over a subset of POAs matched exactly. Reproduce it, find the cause, and add
  a regression test.

_added 2026-11-07_

### Label stacked columns in stacked order and add the families count code

- When `column_variable` is not the last variable in the file name, the reader labels the
  stacked columns in file-name order, so they land on the wrong variable. Label them in
  stacked order, and give the families dataset (`TBF`) and dwellings (`TBD`) count codes
  wherever the column-variable path needs them. Bond Loss works around both with a
  `ColumnLastReader` subclass.

_added 2026-11-07_

### Read ACTEID extracts

- Accept tables from the Australian Census and Temporary Entrants datasets: the title line
  reads "Weighted count of Person level", the geography is labelled "Postal Areas (EN)" or
  "POAUCD Postal Areas (EN)", counts are weighted decimals, each count column is followed by
  RSE and Annotations columns, and a Symbol / Description block precedes the copyright
  lines.

_added 2026-11-07_

## Later / Someday

## Done (last 60 days)
