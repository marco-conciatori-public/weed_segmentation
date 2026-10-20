"""Evaluate a trained Mask R-CNN on a split. Settings: configs/maskrcnn.yaml (`evaluate`,
plus `data`, `run_name` and `output_dir` to find the run and the data).

Writes to <run>/eval/<split>_<checkpoint>/: predictions.json (COCO results) and the outputs
of weedseg.report.write_evaluation: COCO metrics, per-class AP, confusion matrix, previews
and, for species-level classes, taxonomic_levels.csv and (with `evaluate.rules`)
decisions.csv/.json (keep / remove / defer per true role; see weedseg.decisions).
"""

from torch.utils.data import DataLoader

from weedseg.config import load_config
from weedseg.maskrcnn.data import CocoInstanceDataset, collate, load_coco, model_categories
from weedseg.maskrcnn.engine import predict, select_device
from weedseg.maskrcnn.model import load_checkpoint_model
from weedseg.maskrcnn.run import checkpoint_path, eval_dir, run_name, split_file
from weedseg.report import write_evaluation

CONFIG = "configs/maskrcnn.yaml"


def main() -> None:
    config = load_config(CONFIG)
    settings = config["evaluate"]
    checkpoint_file = checkpoint_path(config)
    device = select_device(config["device"])
    amp = config["train"]["amp"] and device.type == "cuda"
    model, checkpoint = load_checkpoint_model(checkpoint_file, device)
    model_config = checkpoint["config"]["model"]
    print(f"{checkpoint_file} (epoch {checkpoint['epoch']}), device {device}")

    split = settings["split"]
    coco = load_coco(split_file(config, split))
    # The model's labels follow the training categories; look them up by name in this file.
    by_name = {c["name"]: c for c in model_categories(coco)}
    missing = [c["name"] for c in checkpoint["categories"] if c["name"] not in by_name]
    if missing:
        raise SystemExit(f"Categories of the model missing from the {split} file: {missing}")
    categories = [by_name[c["name"]] for c in checkpoint["categories"]]

    dataset = CocoInstanceDataset(coco, with_targets=False, max_images=settings["max_images"],
                                  seed=config["seed"])
    if settings["max_images"] is not None:  # evaluate and preview only the chosen images
        image_ids = {i["id"] for i in dataset.images}
        coco = {**coco, "images": dataset.images,
                "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids]}
    loader = DataLoader(dataset, settings["batch_size"], collate_fn=collate,
                        num_workers=config["data"]["num_workers"])
    results = predict(model, loader, categories, device, amp, model_config["mask_threshold"])

    write_evaluation(coco, results, settings, eval_dir(config), run_name(config),
                     model_config["detections_per_image"], dataset.image_dir,
                     split_file(config, split), config["seed"])


if __name__ == "__main__":
    main()
