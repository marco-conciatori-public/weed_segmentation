"""Two-stage prediction and evaluation on a split: the plant detector finds the plants, the
species classifier names them (see weedseg.twostage). Settings: configs/twostage.yaml.

Writes to <output_dir>/<run_name>/eval/<split>/ the same outputs as
scripts/evaluate_maskrcnn.py (weedseg.report.write_evaluation), so the two pipelines can be
compared file by file; scripts/review_errors.py reviews them with `review.eval_dir`.
"""

from torch.utils.data import DataLoader

from weedseg.classifier.model import load_classifier
from weedseg.config import load_config
from weedseg.maskrcnn.data import CocoInstanceDataset, collate, load_coco, model_categories
from weedseg.maskrcnn.engine import predict, select_device
from weedseg.maskrcnn.model import load_checkpoint_model
from weedseg.maskrcnn.run import split_file
from weedseg.paths import resolve
from weedseg.report import write_evaluation
from weedseg.twostage import classify_detections

CONFIG = "configs/twostage.yaml"


def main() -> None:
    config = load_config(CONFIG)
    settings, data = config["evaluate"], config["data"]
    device = select_device(config["device"])
    amp = config["amp"] and device.type == "cuda"
    detector_file = resolve(config["detector"]["checkpoint"])
    classifier_file = resolve(config["classifier"]["checkpoint"])
    detector, detector_checkpoint = load_checkpoint_model(detector_file, device)
    classifier, classifier_checkpoint = load_classifier(classifier_file, device)
    detector_config = detector_checkpoint["config"]["model"]
    print(f"detector {detector_file} (epoch {detector_checkpoint['epoch']}), classifier "
          f"{classifier_file} (epoch {classifier_checkpoint['epoch']}), device {device}")

    split = settings["split"]
    coco = load_coco(split_file(config, split))
    ids = {c["name"]: c["id"] for c in model_categories(coco)}
    classes = classifier_checkpoint["classes"]
    missing = [name for name in classes if name not in ids]
    if missing:
        raise SystemExit(f"Classes of the classifier missing from the {split} file: {missing}")

    dataset = CocoInstanceDataset(coco, with_targets=False, max_images=settings["max_images"],
                                  seed=config["seed"])
    if settings["max_images"] is not None:  # evaluate and preview only the chosen images
        image_ids = {i["id"] for i in dataset.images}
        coco = {**coco, "images": dataset.images,
                "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids]}
    loader = DataLoader(dataset, config["detector"]["batch_size"], collate_fn=collate,
                        num_workers=data["num_workers"])
    plants = predict(detector, loader, detector_checkpoint["categories"], device, amp,
                     detector_config["mask_threshold"])
    results = classify_detections(
        plants, dataset.images, dataset.image_dir, classifier, [ids[name] for name in classes],
        classifier_checkpoint["config"]["crops"], device, amp,
        config["classifier"]["batch_size"], config["classifier"]["min_plant_score"],
        data["num_workers"])

    name = config["run_name"].format(detector=detector_file.parent.name,
                                     classifier=classifier_file.parent.name)
    out_dir = resolve(config["output_dir"]) / name / "eval" / split
    write_evaluation(coco, results, settings, out_dir, name,
                     detector_config["detections_per_image"], dataset.image_dir,
                     split_file(config, split), config["seed"])


if __name__ == "__main__":
    main()
