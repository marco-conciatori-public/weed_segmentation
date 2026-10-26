"""EPPO codes: build a small taxa table from the EPPO SQLite export, and read it back.

The export (`eppocodes_all.sqlite`, ~50 MB, from https://data.eppo.int) is a gitignored
source. `build_taxa` keeps only the codes we use, with their taxonomy and names, so the
table can be versioned and read without the export. See docs/reference-data.md.
"""

import csv
import sqlite3
from collections.abc import Iterable
from pathlib import Path

TAXONOMIC_LINK = 9  # t_link_types: 9 = Taxonomic
# Plant group codes end with a rank letter, e.g. 1CHEG (genus), 1AMAF (family).
RANKS = {"G": "genus", "S": "subfamily", "F": "family", "O": "order", "D": "clade",
         "C": "class", "P": "phylum", "K": "kingdom"}
PARENT_RANKS = ("genus", "family", "order")
# Ranks above order. The taxa table doesn't record which species belong to these.
ABOVE_ORDER = ("clade", "class", "phylum", "kingdom")
NAME_SEPARATOR = "; "
# Optional label-map columns that narrow down a label coarser than species, as EPPO codes
# separated by spaces: the plant belongs to one of the `one_of` taxa and to none of the
# `none_of` taxa (e.g. PhenoBench's weeds: none_of BEAVA). See docs/data-format.md.
LIMIT_COLUMNS = ("one_of", "none_of")
# Ranks whose species the taxa table records (a class is in such a taxon when the taxon is
# its own code, genus, family or order).
LIMIT_RANKS = ("species", *PARENT_RANKS)


def rank_of(code: str, datatype: str) -> str:
    """`species` for PFL codes (species and below), the rank letter for SPT group codes."""
    if datatype == "PFL":
        return "species"
    return RANKS.get(code[-1], "group")


def taxa_columns(languages: Iterable[str]) -> list[str]:
    columns = ["eppo", "datatype", "rank", "active", "name", "synonyms"]
    for rank in PARENT_RANKS:
        columns += [rank, f"{rank}_name"]
    return columns + [f"common_names_{lang}" for lang in languages]


def build_taxa(sqlite_path: Path, codes: Iterable[str],
               languages: Iterable[str] = ("en", "it")) -> list[dict[str, str]]:
    """One row per code: accepted Latin name, Latin synonyms, genus/family/order and
    common names. Codes missing from the export get a row with `active` empty."""
    languages = list(languages)
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        def code_row(eppo: str):
            return con.execute("SELECT codeid, dtcode, status FROM t_codes WHERE eppocode = ?",
                               (eppo,)).fetchone()

        def names(codeid: int, lang: str, preferred: int | None = None) -> list[str]:
            query = ("SELECT fullname FROM t_names WHERE codeid = ? AND codelang = ? "
                     "AND status = 'A'")
            params: tuple = (codeid, lang)
            if preferred is not None:
                query += " AND preferred = ?"
                params += (preferred,)
            return [row[0].strip() for row in con.execute(query + " ORDER BY nameid", params)]

        def parent(codeid: int):
            return con.execute(
                "SELECT k.codeid, k.eppocode, k.dtcode FROM t_links l "
                "JOIN t_codes k ON k.codeid = l.codeid_parent "
                "WHERE l.codeid = ? AND l.idlinkcode = ? AND l.status = 'A'",
                (codeid, TAXONOMIC_LINK)).fetchone()

        rows = []
        for eppo in sorted(set(codes)):
            row = dict.fromkeys(taxa_columns(languages), "")
            row["eppo"] = eppo
            found = code_row(eppo)
            if found is None:
                rows.append(row)
                continue
            codeid, datatype, status = found
            preferred = names(codeid, "la", preferred=1)
            row.update(datatype=datatype, rank=rank_of(eppo, datatype),
                       active="yes" if status == "A" else "no",
                       name=preferred[0] if preferred else "",
                       synonyms=NAME_SEPARATOR.join(names(codeid, "la", preferred=0)))
            for lang in languages:
                row[f"common_names_{lang}"] = NAME_SEPARATOR.join(names(codeid, lang))

            # Walk up the taxonomy; a genus-level code is its own genus.
            current = (codeid, eppo, datatype)
            while current is not None:
                rank = rank_of(current[1], current[2])
                if rank in PARENT_RANKS and not row[rank]:
                    row[rank] = current[1]
                    accepted = names(current[0], "la", preferred=1)
                    row[f"{rank}_name"] = accepted[0] if accepted else ""
                current = parent(current[0])
            rows.append(row)
        return rows
    finally:
        con.close()


def write_taxa(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_taxa(path: Path) -> dict[str, dict[str, str]]:
    """The taxa table as {EPPO code: row}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["eppo"]: row for row in csv.DictReader(f)}


def label_codes(row: dict[str, str]) -> set[str]:
    """Every EPPO code a label-map row names: its `eppo` and its LIMIT_COLUMNS."""
    codes = {row["eppo"]} if row.get("eppo") else set()
    for column in LIMIT_COLUMNS:
        codes.update((row.get(column) or "").split())
    return codes


def known_names(row: dict[str, str]) -> set[str]:
    """Accepted name and Latin synonyms of a taxa row, case-folded."""
    names = [row["name"], *row["synonyms"].split(NAME_SEPARATOR)]
    return {name.casefold() for name in names if name}


def check_label_rows(rows: Iterable[dict[str, str]],
                     taxa: dict[str, dict[str, str]]) -> list[str]:
    """Problems in label-map rows: EPPO codes missing or inactive, a scientific name that is
    neither EPPO's accepted name nor a synonym, or a LIMIT_COLUMNS code above order (the taxa
    table doesn't record which species those hold). Rows with an `eppo_note` explain a
    deliberate mismatch and skip the name check."""
    problems = []
    for row in rows:
        code = row.get("eppo")
        label = row["source_label"]
        for column in LIMIT_COLUMNS:
            for limit in (row.get(column) or "").split():
                taxon = taxa.get(limit)
                if taxon is None or taxon["active"] != "yes":
                    problems.append(f"label {label}: {column} code {limit} is not an active "
                                    "code of the taxa table")
                elif taxon["rank"] not in LIMIT_RANKS:
                    problems.append(f"label {label}: {column} code {limit} is a {taxon['rank']}; "
                                    f"use codes of rank {', '.join(LIMIT_RANKS)}")
            if row.get(column, "").strip() and not code:
                problems.append(f"label {label}: {column} without an EPPO code")
        if not code:
            continue
        taxon = taxa.get(code)
        if taxon is None or not taxon["active"]:
            problems.append(f"label {label}: {code} is not in the EPPO export")
        elif taxon["active"] != "yes":
            problems.append(f"label {label}: {code} is no longer active in EPPO")
        elif row.get("eppo_note"):
            continue
        elif row.get("scientific_name", "").casefold() not in known_names(taxon) | {""}:
            problems.append(f"label {label}: {row['scientific_name']!r} is not an EPPO name "
                            f"of {code} ({taxon['name']})")
    return problems
