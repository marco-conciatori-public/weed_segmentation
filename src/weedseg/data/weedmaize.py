"""Reader for the WeedMaize dataset (López Correa, Andújar et al., CSIC and UPM; Zenodo 5106795):
commercial maize fields in Badajoz, Spain, April 2020, photographed straight down from 1.3 m
with a Canon PowerShot SX540 HS (5184x3888), a box for every plant.

Raw layout after download (see docs/datasets/weedmaize.md):
    {train,test,validation}_set/IMG_<photo>_<tile>.JPG   3000x1296 tiles cut from each photo
    {train,test,validation}_set/IMG_<photo>_<tile>.xml   their Pascal VOC boxes

The tiles of one photo lie on a grid: 6 per row, 400 px apart, 7 rows 400 px apart (tile t
at x = 400 * ((t - 1) % 6), y = 400 * ((t - 1) // 6); checked 2026-11-08 by matching the boxes
neighbouring tiles share). They overlap heavily, so the same plant is in several tiles;
`spread_tiles` keeps a set of tiles that overlap little. Not every tile of the grid is in the
download, and the authors put tiles of one photo in different sets. Image names are
"<set>/IMG_<photo>_<tile>".
"""

import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from weedseg.data.coco import ImageRecord, Instance

SETS = ("train_set", "validation_set", "test_set")
GRID_STEP, GRID_COLUMNS = 400, 6  # the tile grid, see above
NAME = re.compile(r"(?P<set>\w+)/IMG_(?P<photo>\d+)_(?P<tile>\d+)")
Box = tuple[str, tuple[int, int, int, int], bool]  # label, (x0, y0, x1, y1), truncated


def parse(name: str) -> tuple[str, int, int]:
    """(set, photo number, tile number) of an image name."""
    match = NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected WeedMaize image name {name!r}")
    return match["set"], int(match["photo"]), int(match["tile"])


def image_names(raw_dir: Path) -> list[str]:
    names = []
    for folder in SETS:
        if not (raw_dir / folder).is_dir():
            raise FileNotFoundError(f"{raw_dir / folder} missing; run scripts/download_datasets.py")
        names += [f"{folder}/{path.stem}" for path in sorted((raw_dir / folder).glob("IMG_*.xml"))]
    return names


@cache
def read_tile(raw_dir: Path, name: str) -> tuple[tuple[int, int], tuple[Box, ...]]:
    """(width, height) of a tile and its boxes, as its XML gives them (1-based inclusive
    corners turned into 0-based [x0, x1) ranges)."""
    root = ET.parse(raw_dir / f"{name}.xml").getroot()
    size = root.find("size")
    width, height = int(size.findtext("width")), int(size.findtext("height"))
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((obj.findtext("name").strip(), (x0 - 1, y0 - 1, x1, y1),
                      obj.findtext("truncated") == "1"))
    return (width, height), tuple(boxes)


def iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def ioa(inner: tuple[int, ...], outer: tuple[int, ...]) -> float:
    """The share of `inner`'s area that lies in `outer`."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((inner[2] - inner[0]) * (inner[3] - inner[1]))


def drop_duplicates(boxes: tuple[Box, ...], threshold: float) -> list[Box]:
    """The boxes without those that repeat an earlier one: some plants are boxed twice, with
    corners a pixel or two apart. Any label: the second box adds no plant."""
    kept = []
    for box in boxes:
        if all(iou(box[1], other[1]) < threshold for other in kept):
            kept.append(box)
    return kept


def tile_offset(tile: int) -> tuple[int, int]:
    """The top-left corner of a tile (numbered from 1) in its photo, in px."""
    return (GRID_STEP * ((tile - 1) % GRID_COLUMNS), GRID_STEP * ((tile - 1) // GRID_COLUMNS))


def spread_tiles(tiles: list[int], size: tuple[int, int], max_overlap: float) -> list[int]:
    """Those of `tiles` that overlap each other by at most `max_overlap` of a tile's area,
    chosen greedily to cover as much of the photo as possible: each step takes the allowed
    tile that adds the most area not yet covered (the lowest-numbered on a tie). Sorted."""
    width, height = size
    cell = GRID_STEP // 4  # coverage is counted on a grid of 100 px cells
    offsets = {tile: tile_offset(tile) for tile in tiles}
    covered = np.zeros((max(y for _, y in offsets.values()) // cell + height // cell + 1,
                        max(x for x, _ in offsets.values()) // cell + width // cell + 1), bool)

    def window(tile: int) -> tuple[slice, slice]:
        x, y = offsets[tile]
        return slice(y // cell, (y + height) // cell), slice(x // cell, (x + width) // cell)

    def allowed(tile: int, kept: list[int]) -> bool:
        x, y = offsets[tile]
        return all(max(0, width - abs(x - offsets[k][0])) * max(0, height - abs(y - offsets[k][1]))
                   <= max_overlap * width * height for k in kept)

    kept = []
    while True:
        gains = [(int((~covered[window(tile)]).sum()), -tile) for tile in sorted(offsets)
                 if tile not in kept and allowed(tile, kept)]
        if not gains or max(gains)[0] == 0:
            return sorted(kept)
        tile = -max(gains)[1]
        kept.append(tile)
        covered[window(tile)] = True


def select_tiles(raw_dir: Path, names: list[str], max_overlap: float) -> list[str]:
    """The names of the tiles `spread_tiles` keeps, photo by photo, whatever set each tile is
    in."""
    by_photo = defaultdict(dict)
    for name in names:
        _, photo, tile = parse(name)
        by_photo[photo][tile] = name
    kept = []
    for photo, tiles in sorted(by_photo.items()):
        size = read_tile(raw_dir, next(iter(tiles.values())))[0]
        kept += [tiles[tile] for tile in spread_tiles(list(tiles), size, max_overlap)]
    return sorted(kept)


def group_of(name: str, block: int) -> str:
    """The block of `block` consecutive photo numbers the tile's photo belongs to."""
    _, photo, _ = parse(name)
    start = photo // block * block
    return f"IMG_{start:04d}-{start + block - 1:04d}"


@dataclass(frozen=True)
class Labelling:
    """How the tiles' boxes become plants. `duplicate_iou`: a box whose IoU with an earlier one
    reaches this repeats it and is dropped. `centres`: {label: label of the whole plant}, for
    plants boxed twice, the whole plant and its centre (CYPRO_min inside CYPRO_max): a centre
    box at least `centre_ioa` inside a whole-plant box is dropped; one inside none marks a
    plant whose extent wasn't drawn, an ignore region."""

    duplicate_iou: float = 0.9
    centres: dict[str, str] = field(default_factory=dict)
    centre_ioa: float = 0.5

    def __hash__(self) -> int:
        return hash((self.duplicate_iou, tuple(sorted(self.centres.items())), self.centre_ioa))


def plants(raw_dir: Path, name: str, labelling: Labelling) -> list[tuple[Box, bool]]:
    """The boxes of a tile as plants: (box, ignore), duplicates and paired centres dropped."""
    boxes = drop_duplicates(read_tile(raw_dir, name)[1], labelling.duplicate_iou)
    result = []
    for box in boxes:
        whole = labelling.centres.get(box[0])
        if whole is None:
            result.append((box, False))
        elif not any(other[0] == whole and ioa(box[1], other[1]) >= labelling.centre_ioa
                     for other in boxes):
            result.append((box, True))
    return result


def box_labels(raw_dir: Path, name: str, labelling: Labelling) -> list[tuple[str, int, int]]:
    """(source label, width, height) of each plant of a tile, in px (ignore regions left
    out)."""
    return [(label, x1 - x0, y1 - y0)
            for (label, (x0, y0, x1, y1), _), ignore in plants(raw_dir, name, labelling)
            if not ignore]


def iter_records(raw_dir: Path, names: list[str], block: int, labelling: Labelling,
                 mm_per_px: float | None = None) -> Iterator[ImageRecord]:
    """Yield one ImageRecord per tile name, labels as the dataset's class names (EPPO code
    plus growth stage, e.g. ZEAMX_V1), see `plants`. Annotation fields: `truncated` for plants
    cut by the tile's edge, `centre_only` for a centre box without its whole plant (an ignore
    region). Image fields: `photo`, `tile`, `set` (the authors' split), `mm_per_px`
    (estimated)."""
    for name in names:
        folder, photo, tile = parse(name)
        width, height = read_tile(raw_dir, name)[0]
        attributes = {"photo": photo, "tile": tile, "set": folder}
        if mm_per_px is not None:
            attributes["mm_per_px"] = mm_per_px
            attributes["mm_per_px_estimated"] = True
        instances = []
        for (label, box, truncated), ignore in plants(raw_dir, name, labelling):
            extra = {"truncated": True} if truncated else {}
            if ignore:
                extra["centre_only"] = True
            instances.append(Instance(source_label=label, bbox_xyxy=box, ignore=ignore,
                                      attributes=extra))
        yield ImageRecord(file_name=f"{name}.JPG", width=width, height=height,
                          group=group_of(name, block), instances=instances,
                          attributes=attributes)
