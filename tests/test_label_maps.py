"""Checks the versioned label maps against the versioned EPPO taxa table."""

import csv

import pytest

from weedseg.eppo import check_label_rows, load_taxa
from weedseg.paths import resolve

TAXA = load_taxa(resolve("configs/eppo_taxa.csv"))
LABEL_MAPS = sorted(resolve("configs/label_maps").glob("*.csv"))


@pytest.mark.parametrize("path", LABEL_MAPS, ids=lambda p: p.name)
def test_label_map_matches_eppo(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert check_label_rows(rows, TAXA) == []
    for row in rows:
        if row["eppo"]:
            assert row["role"] in {"crop", "weed"}
