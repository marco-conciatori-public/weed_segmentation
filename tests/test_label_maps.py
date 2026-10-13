"""Checks the versioned label maps against the versioned EPPO taxa table."""

import csv

import pytest

from weedseg.data.coco import IGNORE_CATEGORY, LabelMap
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


def test_genus_taxonomy_from_eppo():
    label_map = LabelMap(resolve("configs/label_maps/cropandweed.csv"), "genus", TAXA)
    assert label_map.category("33") == "1CHEG"  # Chenopodium album
    assert label_map.category("65") == label_map.category("68") == "1POAG"  # two Poa species
    assert label_map.category("255") == IGNORE_CATEGORY


@pytest.mark.parametrize("rank", ["genus", "family", "order"])
def test_rank_taxonomies_never_mix_crops_and_weeds(rank):
    label_map = LabelMap(resolve("configs/label_maps/cropandweed.csv"), rank, TAXA)
    assert label_map.category("1") == "ZEAMX"  # maize stays a species
    assert all(c["supercategory"] != "mixed" for c in label_map.categories())
