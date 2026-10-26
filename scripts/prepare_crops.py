"""Cut every labelled plant of a species dataset into the classifier's crop cache.
Settings: configs/classifier.yaml (`data`, `crops`). CPU only.

Writes <data.dir>/crops_margin<m>_size<s>_context<c>/<split>/ (<annotation id>.jpg and .png,
the plant's mask) and index_<split>.csv. A split whose index lists the plants of the current
data is skipped; one made from other data (e.g. before re-combining) is cut again. Delete the
index to redo a split anyway.
"""

from weedseg.classifier.crops import cache_dir, index_is_current, write_cache
from weedseg.config import load_config
from weedseg.data.combine import SPLITS
from weedseg.maskrcnn.data import load_coco
from weedseg.maskrcnn.run import split_file
from weedseg.paths import resolve

CONFIG = "configs/classifier.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    folder = cache_dir(resolve(config["data"]["dir"]), config["crops"])
    for split in SPLITS:
        index = folder / f"index_{split}.csv"
        coco = load_coco(split_file(config, split))
        if index.exists():
            if index_is_current(index, coco):
                print(f"{split}: {index} is up to date, skipped")
                continue
            print(f"{split}: {index} was made from other data, cutting the crops again")
        write_cache(coco, resolve(coco["info"]["image_dir"]), folder / split, config["crops"],
                    config["data"]["num_workers"])
        print(f"{split} -> {index}")


if __name__ == "__main__":
    main()
