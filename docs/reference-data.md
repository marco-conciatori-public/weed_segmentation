# Reference data

## EPPO codes

Species are identified by EPPO codes (e.g. `CHEAL` = *Chenopodium album*), see
[data-format.md](data-format.md). This project keeps a small table of the codes it uses,
built from EPPO's full export.

| | |
|---|---|
| Source | EPPO Global Database, full export `eppocodes_all.sqlite` from the EPPO Data Portal, <https://data.eppo.int> (free account) |
| Export used | downloaded 2026-06-24, copied from the AgronomyProceedingsExtraction project |
| Licence | EPPO Codes Open Data Licence (2014-11-17): reuse, including commercial, is allowed **with attribution** |
| Raw export | `data/ref/sources/eppocodes_all.sqlite` (~50 MB, gitignored) |
| Our table | `configs/eppo_taxa.csv` (versioned) |
| Built by | `scripts/build_eppo_taxa.py`, settings in `configs/eppo.yaml` |

### Required attribution

The licence asks for EPPO's name and the date the codes were last downloaded, without
suggesting any endorsement by EPPO and without the EPPO logo. Use this wherever results
built on the codes are published (papers, datasets, model cards):

> Contains EPPO Codes (www.eppo.int), downloaded 2026-06-24.

Update the date when the export is refreshed.

### The taxa table

One row per EPPO code used by any label map (plus `extra_codes` from the config):

| Column | Meaning |
|---|---|
| `eppo`, `datatype`, `rank`, `active` | the code; `PFL` = plant species, `SPT` = plant group; rank from the code (e.g. `1CHEG` ends in G = genus); whether EPPO still uses it |
| `name`, `synonyms` | EPPO's accepted Latin name and its Latin synonyms |
| `genus`, `family`, `order` (+ `_name`) | the code's taxonomic parents. A genus code is its own genus. |
| `common_names_en`, `common_names_it` | vernacular names. These are all the active ones, in EPPO's order, so the first is not necessarily the most common. |

### Rebuilding

Run `scripts/build_eppo_taxa.py`, needed after adding codes to a label map or downloading
a newer export. It rewrites the table and then checks every label map against it,
printing:
- codes that are missing from EPPO or no longer active,
- scientific names in a label map that are neither EPPO's accepted name nor a synonym.

A deliberate mismatch (e.g. the dataset names an aggregate and we use the species code)
is explained in the label map's `eppo_note` column, which skips the name check for that
row. The test `tests/test_label_maps.py` runs the same check on the versioned files, so
it also works without the raw export.

### Why a small table and not the whole export

We only map a few curated labels, not free text, so we don't need the full name index
with fuzzy matching that AgronomyProceedingsExtraction builds. A small versioned table
keeps changes reviewable in git, and training never needs the 50 MB export.
