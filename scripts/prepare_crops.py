"""Cut every labelled plant of a species dataset into the classifier's crop cache.
Settings: configs/classifier.yaml (`data`, `crops`). CPU only.

Writes <data.dir>/crops_margin<m>_size<s>_context<c>/<split>/ (<annotation id>.jpg and .png,
the plant's mask) and index_<split>.csv. Splits whose index exists are skipped; delete the
index to redo one.
"""

from weedseg.classifier.crops import cache_dir, write_cache
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
        if index.exists():
            print(f"{split}: {index} exists, skipped")
            continue
        coco = load_coco(split_file(config, split))
        write_cache(coco, resolve(coco["info"]["image_dir"]), folder / split, config["crops"],
                    config["data"]["num_workers"])
        print(f"{split} -> {index}")


if __name__ == "__main__":
    main()
