"""Score classifier runs on plants of datasets they never trained on, next to runs that did.
Settings: configs/evaluate_unseen.yaml. See weedseg.unseen for what is measured, and
docs/twostage.md (Unseen sites).

Cuts only the crops needed, to <output_dir>/crops/ (skipped when up to date), and keeps each
run's probabilities in <output_dir>/<run>/<split>_<checkpoint>.npz, so a stopped session
goes on where it was and only the table is made again. Writes
<output_dir>/unseen_<date>_<time>.md and .csv, and prints the table.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.data import DataLoader

from weedseg.classifier.crops import index_is_current, read_index, write_cache
from weedseg.classifier.data import CropDataset
from weedseg.classifier.engine import predict_probs
from weedseg.classifier.model import load_classifier
from weedseg.compare import run_dir
from weedseg.config import load_config
from weedseg.data.coco import IGNORE_CATEGORY
from weedseg.decisions import load_rules, role_of
from weedseg.eppo import load_taxa
from weedseg.maskrcnn.data import load_coco
from weedseg.maskrcnn.engine import select_device
from weedseg.paths import resolve
from weedseg.unseen import score_groups, scored_on, site_of, subset, trained_on, write_results

CONFIG = "configs/evaluate_unseen.yaml"


def load_plants(config: dict[str, Any], crops: dict[str, Any],
                folder: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The plants to score, cut into `folder` if needed: index rows with `split` and `site`,
    in the order of the splits. Also the data's class names."""
    splits = sorted({split for wanted in config["datasets"].values() for split in wanted})
    plants, data_classes = [], []
    for split in splits:
        names = [name for name, wanted in config["datasets"].items() if split in wanted]
        coco = subset(load_coco(resolve(config["data_dir"]) / f"{split}.json"), names)
        data_classes = [c["name"] for c in coco["categories"] if c["name"] != IGNORE_CATEGORY]
        index = folder / f"index_{split}.csv"
        if not index.exists() or not index_is_current(index, coco):
            write_cache(coco, resolve(coco["info"]["image_dir"]), folder / split, crops,
                        config["num_workers"])
        images = {image["id"]: image for image in coco["images"]}
        for row in read_index(index):
            plants.append(row | {"split": split,
                                 "site": site_of(images[row["image_id"]], config["sites"])})
    return plants, data_classes


def predict_split(checkpoint_file: Path, rows: list[dict[str, Any]], folder: Path,
                  config: dict[str, Any], device) -> tuple[np.ndarray, list[str]]:
    """The run's probabilities on these plants of one split, and its classes. Scoring goes by
    class name, so the labels CropDataset gives are unused, and all plants pass as its first
    class: a plant may be of a species the run has no class for, which it would refuse."""
    model, checkpoint = load_classifier(checkpoint_file, device)
    classes, crops = checkpoint["classes"], checkpoint["config"]["crops"]
    del checkpoint  # last.pt holds the optimiser too: about 1 GB
    dataset = CropDataset([{**row, "category": classes[0], "allowed": []} for row in rows],
                          folder, classes, crops)
    loader = DataLoader(dataset, config["batch_size"], num_workers=config["num_workers"])
    probs, _ = predict_probs(model, loader, device, device.type == "cuda")
    return probs, classes


def run_probs(run: Path, split: str, rows: list[dict[str, Any]], folder: Path,
              config: dict[str, Any], device, out_dir: Path) -> tuple[np.ndarray, list[str]]:
    """`predict_split`, kept in <out_dir>/<run>/<split>_<checkpoint>.npz for the same plants."""
    path = out_dir / run.name / f"{split}_{config['checkpoint']}.npz"
    ids = np.array([row["annotation_id"] for row in rows])
    if path.exists():
        saved = np.load(path)
        if np.array_equal(saved["annotation_ids"], ids):
            return saved["probs"], saved["classes"].tolist()
        print(f"{path} holds other plants: predicting again")
    print(f"{run.name}, {split}: {len(rows)} plants")
    probs, classes = predict_split(run / f"{config['checkpoint']}.pt", rows, folder, config,
                                   device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, annotation_ids=ids, probs=probs, classes=np.array(classes))
    return probs, classes


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    device = select_device(config["device"])
    out_dir = resolve(config["output_dir"])
    taxa = load_taxa(resolve(config["eppo_taxa"]))
    rules = load_rules(resolve(config["rules"]), taxa)
    if rules.defer_below is None:
        raise SystemExit(f"{config['rules']} has no defer_below")

    runs = [run_dir(entry) for entry in config["runs"]]
    settings = [yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) for run in runs]
    if any(s["crops"] != settings[0]["crops"] for s in settings):
        raise SystemExit("The runs use different crop settings; one crop cache can't serve them")
    plants, data_classes = load_plants(config, settings[0]["crops"], out_dir / "crops")

    results = []
    for run, run_settings in zip(runs, settings):
        datasets = trained_on(resolve(run_settings["data"]["dir"]))
        scores = {}
        for split in sorted({row["split"] for row in plants}):
            rows = [row for row in plants if row["split"] == split
                    and scored_on(split, row["dataset"], datasets)]
            if not rows:
                continue
            probs, classes = run_probs(run, split, rows, out_dir / "crops" / split, config,
                                       device, out_dir)
            roles = {name: role_of(name, rules, taxa) for name in classes}
            scores |= score_groups(rows, probs, classes, roles, taxa, rules.defer_below,
                                   config["sure"], data_classes)
        results.append({"name": run.name, "datasets": datasets, "classes": classes,
                        "scores": scores})
    stem = out_dir / f"unseen_{datetime.now():%Y%m%d_%H%M%S}"
    print(write_results(results, config["sure"], rules.defer_below, stem))
    print(f"-> {stem}.md, {stem.name}.csv")


if __name__ == "__main__":
    main()
