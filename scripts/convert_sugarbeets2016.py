"""Convert the first part of Sugar Beets 2016 to COCO, as a test set only (test.json). Settings:
configs/sugarbeets2016.yaml (convert).

Unpacks the downloaded RAR part with 7-Zip first, if not done yet. Each connected patch of the
crop and weed masks becomes a box, so touching beets merge and some weeds split: the set shows
how the detector and imperfect labels behave, it is not for training or for the detector's
headline numbers. The pipeline (size filter, COCO files) is weedseg.data.convert.
"""

from weedseg.config import load_config
from weedseg.data import sugarbeets2016 as sb
from weedseg.data.coco import LabelMap
from weedseg.data.convert import convert_dataset
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

CONFIG = "configs/sugarbeets2016.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    raw_dir = resolve(config["raw_dir"])
    settings = config["convert"]
    label_map = LabelMap(resolve(settings["label_map"]), settings["taxonomy"],
                         load_taxa(resolve(settings["eppo_taxa"])))
    try:
        sb.unpack(raw_dir, settings["archive"], settings["seven_zip"])
        names = sb.image_names(raw_dir)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    classes = tuple((label, tuple(values)) for label, values in settings["classes"].items())
    min_px = settings["min_patch_px"]
    print(f"{len(names)} images with an index map")

    def records_of(split_names: list[str]):
        return sb.iter_records(raw_dir, split_names, classes, min_px,
                               settings["mm_per_px_estimate"])

    convert_dataset(names, sb.group_of, lambda name: sb.box_labels(raw_dir, name, classes, min_px),
                    records_of, label_map, settings, info={
                        "name": "Sugar Beets 2016 (part 1)",
                        "source": "https://www.ipb.uni-bonn.de/data/sugarbeets2016/",
                        "licence": "CC BY-SA 4.0; see docs/datasets/sugarbeets2016.md",
                        # from the project root
                        "image_dir": f"{config['raw_dir']}/{sb.ROOT}",
                    }, splits={"test": names})


if __name__ == "__main__":
    main()
