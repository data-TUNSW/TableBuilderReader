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
