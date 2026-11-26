"""Build configs/eppo_taxa.csv from the EPPO SQLite export. Settings: configs/eppo.yaml.

Re-run after adding codes to a label map, or after downloading a newer EPPO export. It also
checks every label map against the new table and prints any problem.
"""

import csv

from weedseg.config import load_config
from weedseg.eppo import build_taxa, check_label_rows, label_codes, write_taxa
from weedseg.paths import resolve

CONFIG = "configs/eppo.yaml"


def main() -> None:
    config = load_config(CONFIG)
    sqlite_path = resolve(config["sqlite"])
    if not sqlite_path.exists():
        raise SystemExit(f"EPPO export not found: {sqlite_path}. Download eppocodes_all.sqlite "
                         "from https://data.eppo.int (see docs/reference-data.md).")

    label_maps = {}
    for path in sorted(resolve(config["label_maps"]).glob("*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            label_maps[path.name] = list(csv.DictReader(f))
    codes = {code for rows in label_maps.values() for row in rows for code in label_codes(row)}

    rows = build_taxa(sqlite_path, codes | set(config["extra_codes"]), config["languages"])
    output = resolve(config["output"])
    write_taxa(rows, output)
    print(f"Wrote {len(rows)} codes to {output}")

    taxa = {row["eppo"]: row for row in rows}
    for name, label_rows in label_maps.items():
        problems = check_label_rows(label_rows, taxa)
        print(f"{name}: {len(problems) or 'no'} problem(s)")
        for problem in problems:
            print(f"  {problem}")


if __name__ == "__main__":
    main()
