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

## To do

### Add POA-to-GCCSA weightings

- Register a POA x GCCSA pair in `weightings_registry.py` and generate the combined
  weightings file for it (Census 2021; 2016 too if the registry supports it), so
  postcode data can be reallocated to Greater Sydney and the rest of each state. The
  `*_POA_GCCSA_*` TableBuilder extracts for both Censuses are already in the raw folders.
  Requested by the Bond Loss cluster demographics work, which needs a Greater Sydney
  comparator.

_added 2026-11-06_

## Later / Someday

## Done (last 60 days)
