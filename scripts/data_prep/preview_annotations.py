"""Draw the annotations of a COCO file over a random sample of its images, for a visual check.
Settings: configs/preview.yaml.

Grey hatched areas are ignore regions (iscrowd); crosses mark stem positions.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from weedseg.config import load_config
from weedseg.paths import resolve
from weedseg.visualize import draw_annotations, put_label

CONFIG = "configs/preview.yaml"


def main() -> None:
    config = load_config(CONFIG)
    coco_path = resolve(config["coco_json"])
    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)
    image_dir = resolve(coco["info"]["image_dir"])
    output_dir = (resolve(config["output_dir"]) / coco_path.parent.parent.name
                  / coco_path.parent.name / coco_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    images = [image for image in coco["images"]
              if by_image[image["id"]] or not config["only_annotated"]]
    sample = random.Random(config["seed"]).sample(images, min(config["num_images"], len(images)))

    draw = config["draw"]
    for image_info in tqdm(sample, desc="preview"):
        image = cv2.imread(str(image_dir / image_info["file_name"]))
        if image is None:
            raise FileNotFoundError(image_dir / image_info["file_name"])
        annotations = by_image[image_info["id"]]
        result = draw_annotations(image, annotations, coco["categories"], draw["mask_alpha"],
                                  draw["boxes"], draw["labels"], draw["stems"])
        crowd = sum(a["iscrowd"] for a in annotations)
        put_label(result, f"{image_info['file_name']}  {len(annotations) - crowd} instances, "
                          f"{crowd} ignore", (0, 0), (0, 0, 0), 0.7, 2)
        stem = Path(image_info["file_name"]).stem
        cv2.imwrite(str(output_dir / f"{stem}.jpg"), result, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if config["save_original"]:
            cv2.imwrite(str(output_dir / f"{stem}_original.jpg"), image)
    print(f"{len(sample)} previews -> {output_dir}")


if __name__ == "__main__":
    main()
