"""Write a one-class (plant) copy of a species dataset, for the two-stage detector.
Settings: configs/plant_dataset.yaml. See weedseg.data.plant for which ignore regions become
plants.

Writes train.json, val.json and test.json to a sibling of source_dir named
plant[_min<size>mm]. Images labelled with boxes only keep their mark: the detector's
training leaves them out (see weedseg.maskrcnn.data.load_masked_coco).
"""

from weedseg.config import load_config
from weedseg.data.coco import BOX_ONLY, write_json
from weedseg.data.combine import SPLITS
from weedseg.data.plant import to_plant_only
from weedseg.maskrcnn.data import load_coco
from weedseg.paths import resolve

CONFIG = "configs/plant_dataset.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    source_dir = resolve(config["source_dir"])
    output_dir = None
    for split in SPLITS:
        species = load_coco(source_dir / f"{split}.json")
        plants = to_plant_only(species)
        if output_dir is None:
            size = plants["info"].get("min_plant_size_mm")
            output_dir = source_dir.parent / ("plant" + ("" if size is None else f"_min{size:g}mm"))
        write_json(plants, output_dir / f"{split}.json")
        before = sum(1 for a in species["annotations"] if not a.get("iscrowd"))
        after = sum(1 for a in plants["annotations"] if not a["iscrowd"])
        box_only = sum(1 for image in plants["images"] if image.get(BOX_ONLY))
        print(f"{split}: {len(plants['images'])} images"
              + (f" ({box_only} with boxes only, left out by the detector)" if box_only else "")
              + f", {after} plants ({after - before} hidden by the choice of classes)"
              f" -> {output_dir}")


if __name__ == "__main__":
    main()
