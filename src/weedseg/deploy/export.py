"""Deployment package of the two-stage model: everything needed to run it outside this repo.

<output_dir>/<name>/ gets the weights of both stages as safetensors (plain tensors, no
pickled code), model.json (the settings weed_model.py needs, the species list and the
hand-off thresholds), weed_model.py and requirements.txt as they are, splits.csv (every
image of the training data and the split each model put it in), and README.md, made from
README_template.md with the species, training data and validation numbers of the runs.
See docs/twostage.md#deployment-package.
"""

import csv
import hashlib
import json
import platform
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median
from string import Template
from typing import Any

import timm
import torch
import torchvision
from safetensors.torch import save_file
from tqdm import tqdm

from weedseg.classifier.crops import FILL_RGB
from weedseg.classifier.data import MEAN, STD
from weedseg.config import load_config
from weedseg.data.coco import BOX_ONLY, IGNORE_CATEGORY, OTHER_CATEGORY
from weedseg.eppo import NAME_SEPARATOR, load_taxa
from weedseg.maskrcnn.run import checkpoint_path
from weedseg.paths import resolve

HERE = Path(__file__).resolve().parent
COPIED = ("weed_model.py", "requirements.txt")
SPLITS = ("train", "val", "test")
# Papers to cite, per dataset id of the combined data (see docs/datasets/).
PAPERS = {
    "cropandweed": "Steininger et al., \"The CropAndWeed Dataset: a Multi-Modal Learning "
                   "Approach for Efficient Crop and Weed Manipulation\", WACV 2023",
    "acre": "\"The ACRE Crop-Weed Dataset for Benchmarking Weed Detection Models on Maize "
            "and Beans Fields\", ECML PKDD 2023 workshops (CCIS, 2025)",
    "phenobench": "Weyler et al., \"PhenoBench: A Large Dataset and Benchmarks for Semantic "
                  "Image Interpretation in the Agricultural Domain\", IEEE TPAMI 2024",
    "oppd": "Madsen et al., \"Open Plant Phenotyping Database of Common Weeds in Denmark\", "
            "Remote Sensing 2020",
    "groundbased_weed": "Moreno, Rivera Cárdenas and Andújar, \"Ground-based imagery dataset "
                        "for early weed classification in tomato crops\", Data in Brief 2025",
}
# Classifier backbones by a part of their timm name: the name in the README, the licence of
# the pretrained weights, the licence file that must go with them (and with the fine-tuned
# ones; None: none) and the terms the README states. An unknown backbone stops the export,
# so its licence is looked up first.
BACKBONES = {
    "dinov3": {
        "name": "DINOv3", "licence": "Meta's DINOv3 License", "file": "DINOv3_LICENSE.md",
        # the licence's section 1.b, as https://ai.meta.com/resources/models-and-libraries/
        # dinov3-license states it (the model card of the weights links there)
        "terms": "**Built with DINOv3.** The classifier is a fine-tuned DINOv3 model, so its "
                 "weights (`classifier.safetensors`) come under the DINOv3 License, a copy of "
                 "which is `DINOv3_LICENSE.md`; the licence itself is what counts. In short: "
                 "pass the licence on with the weights, and the weights only under its terms; "
                 "show \"Built with DINOv3\" prominently on a related website, user interface, "
                 "blog post, about page or product documentation; acknowledge DINOv3 in "
                 "publications; and don't use the model against trade controls or for military, "
                 "warfare, nuclear, espionage or weapons purposes.",
    },
    "dinov2": {"name": "DINOv2", "licence": "Apache-2.0", "file": None, "terms": ""},
}


def export_package(config: dict[str, Any]) -> Path:
    """Write the package that `config` (configs/export_twostage.yaml) describes; its folder."""
    twostage = load_config(config["twostage"])
    detector_config = load_config(twostage["detector"]["config"])
    classifier_config = load_config(twostage["classifier"]["config"])
    detector_file = checkpoint_path(detector_config, twostage["detector"]["checkpoint"])
    classifier_file = checkpoint_path(classifier_config, twostage["classifier"]["checkpoint"])
    detector = torch.load(detector_file, map_location="cpu", weights_only=False)
    classifier = torch.load(classifier_file, map_location="cpu", weights_only=False)
    if [c["name"] for c in detector["categories"]] != ["plant"]:
        raise SystemExit(f"{detector_file} is not a plant detector (one class, plant)")
    crops = classifier["config"]["crops"]
    if crops["background"] != "keep":
        raise SystemExit(f"weed_model.py only cuts crops with background 'keep', not "
                         f"{crops['background']!r}")
    backbone = backbone_family(classifier["config"]["model"]["backbone"])

    created = date.today().isoformat()
    out_dir = resolve(config["output_dir"]) / config["name"].format(date=created)
    if out_dir.exists():
        if not config["overwrite"]:
            raise SystemExit(f"{out_dir} exists; set overwrite: true to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    taxa = load_taxa(resolve(config["eppo_taxa"]))
    species_dir = resolve(classifier["config"]["data"]["dir"])
    train = json.loads((species_dir / "train.json").read_text(encoding="utf-8"))
    detector_settings = detector["config"]["model"]
    info = {
        "name": out_dir.name,
        "created": created,
        "detector": {
            "architecture": "torchvision maskrcnn_resnet50_fpn_v2, one class (plant)",
            # several sizes are a training augmentation; evaluation uses the last
            "min_size": detector_settings["min_size"][-1],
            "max_size": detector_settings["max_size"],
            "anchor_sizes": detector_settings["anchor_sizes"],
            "detections_per_image": detector_settings["detections_per_image"],
            "score_threshold": detector_settings["score_threshold"],
            "mask_threshold": detector_settings["mask_threshold"],
            "run": detector_file.parent.name, "epoch": detector["epoch"],
        },
        "classifier": {
            "backbone": classifier["config"]["model"]["backbone"],
            "backbone_licence": backbone["licence"] + (f" ({backbone['file']})"
                                                       if backbone["file"] else ""),
            "input_size": crops["input_size"], "margin": crops["margin"],
            "fill_rgb": list(FILL_RGB), "mean": MEAN.flatten().tolist(),
            "std": STD.flatten().tolist(), "batch_size": twostage["classifier"]["batch_size"],
            "run": classifier_file.parent.name, "epoch": classifier["epoch"],
            "partial_labels": classifier["config"]["data"].get("partial_labels", False),
        },
        "handoff": config["handoff"],
        "min_plant_size_mm": train["info"]["min_plant_size_mm"],
        "classes": [describe_class(code, taxa) for code in classifier["classes"]],
        "training_data": {key: {"name": d["name"], "source": d["source"],
                                "licence": d["licence"].split("; see ")[0]}
                          for key, d in train["info"]["datasets"].items()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "timm": timm.__version__},
    }
    (out_dir / "model.json").write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    for name, checkpoint in (("detector", detector), ("classifier", classifier)):
        save_file({k: v.contiguous() for k, v in checkpoint["model"].items()},
                  out_dir / f"{name}.safetensors", metadata={"run": info[name]["run"]})
    for name in COPIED + ((backbone["file"],) if backbone["file"] else ()):
        shutil.copyfile(HERE / name, out_dir / name)
    splits = write_splits(out_dir / "splits.csv",
                          {"detector": resolve(detector["config"]["data"]["dir"]),
                           "classifier": species_dir})

    twostage_eval = (resolve(twostage["output_dir"])
                     / twostage["run_name"].format(detector=detector_file.parent.name,
                                                   classifier=classifier_file.parent.name)
                     / "eval" / f"{twostage['evaluate']['split']}_{classifier_file.stem}")
    # as evaluate_classifier.py names it (weedseg.maskrcnn.run.eval_dir), next to the checkpoint
    classifier_eval = (classifier_file.parent / "eval"
                       / f"{classifier_config['evaluate']['split']}_{classifier_file.stem}")
    readme = readme_text(info, train, classifier["classes"], taxa, twostage_eval,
                         classifier_eval, config["handoff_table"], splits)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"Package ({size:.0f} MB): {out_dir}")
    return out_dir


def backbone_family(backbone: str) -> dict[str, Any]:
    """The BACKBONES entry of a timm model name."""
    for key, family in BACKBONES.items():
        if key in backbone:
            return family
    raise SystemExit(f"No licence known for the backbone {backbone}: look it up and add it "
                     f"to BACKBONES in {Path(__file__).name}")


def describe_class(code: str, taxa: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Code, scientific name, English names (EPPO's, which marks none as preferred) and
    family of a class."""
    if code == OTHER_CATEGORY:
        return {"code": code, "name": "Other plant (a species not in this list)",
                "common_names": [], "family": ""}
    row = taxa[code]
    names = [" ".join(name.split()) for name in row["common_names_en"].split(NAME_SEPARATOR)]
    return {"code": code, "name": row["name"], "common_names": [n for n in names if n],
            "family": row["family_name"]}


def readme_text(info: dict[str, Any], train: dict[str, Any], classes: list[str],
                taxa: dict[str, dict[str, str]], twostage_eval: Path, classifier_eval: Path,
                thresholds: list[float], splits: list[dict[str, str]]) -> str:
    """README_template.md filled in; `splits`: the rows of splits.csv."""
    datasets = train["info"]["datasets"]
    backbone = backbone_family(info["classifier"]["backbone"])
    values = {
        "name": info["name"], "created": info["created"],
        "detector_run": info["detector"]["run"], "detector_epoch": info["detector"]["epoch"],
        "classifier_run": info["classifier"]["run"],
        "classifier_epoch": info["classifier"]["epoch"],
        "backbone": info["classifier"]["backbone"],
        "backbone_family": backbone["name"],
        "backbone_licence": backbone["licence"],
        # each placeholder below is empty for a backbone without licence terms
        "built_with": (f"\n**Built with {backbone['name']}** (see [Licences](#licences)).\n"
                       if backbone["terms"] else ""),
        "licence_file_row": (f"\n| `{backbone['file']}` | The licence of the classifier's "
                             f"weights, which must go with them |" if backbone["file"] else ""),
        "backbone_terms": f"\n{backbone['terms']}\n" if backbone["terms"] else "",
        "num_species": len(classes) - (OTHER_CATEGORY in classes),
        "min_size": info["detector"]["min_size"], "max_size": info["detector"]["max_size"],
        "min_plant_size_mm": info["min_plant_size_mm"],
        **{key: value for key, value in info["handoff"].items()},
        **{key: value for key, value in info["versions"].items()},
        "species_table": species_table(train, classes, taxa, datasets, classifier_eval),
        "other_plant_taxa": other_plant_taxa(train, taxa),
        "handoff_table": handoff_table(classifier_eval, thresholds),
        "results_table": results_table(twostage_eval, datasets),
        "classifier_table": classifier_table(classifier_eval, datasets),
        "partial_section": partial_section(classifier_eval, datasets, taxa,
                                           info["classifier"]["partial_labels"]),
        "dataset_table": dataset_table(train),
        "split_table": split_table(splits),
    }
    template = Template((HERE / "README_template.md").read_text(encoding="utf-8"))
    return template.substitute(values)


def species_table(train: dict[str, Any], classes: list[str], taxa: dict[str, dict[str, str]],
                  datasets: dict[str, Any], classifier_eval: Path) -> str:
    """One row per class: names, training plants per dataset, val accuracy of the
    classifier."""
    counts = plants_per_class(train)
    accuracy = {row["category"]: row for row in read_csv(classifier_eval / "per_class.csv")}
    rows = ["| Code | Species | English names | Family | Training plants | Val accuracy |",
            "|---|---|---|---|---|---|"]
    for code in classes:
        described = describe_class(code, taxa)
        sources = ", ".join(f"{datasets[d]['name']} {n:,}"
                            for d, n in sorted(counts[code].items(), key=lambda kv: -kv[1]))
        row = accuracy.get(code)
        val = (f"{float(row['accuracy']):.0%} of {int(row['crops']):,}"
               if row and row["accuracy"] else "no val plants")
        name = described["name"] if code == OTHER_CATEGORY else f"*{described['name']}*"
        rows.append(f"| `{code}` | {name} | {', '.join(described['common_names'][:3])} | "
                    f"{described['family']} | {sources} | {val} |")
    return "\n".join(rows)


def plants_per_class(train: dict[str, Any]) -> dict[str, Counter]:
    """Labelled training plants per class and dataset (box-only datasets included: the
    classifier learns from them)."""
    dataset_of = {image["id"]: image.get("dataset") for image in train["images"]}
    names = {c["id"]: c["name"] for c in train["categories"]}
    counts = defaultdict(Counter)
    for annotation in train["annotations"]:
        name = names[annotation["category_id"]]
        if name != IGNORE_CATEGORY and not annotation.get("iscrowd"):
            counts[name][dataset_of[annotation["image_id"]]] += 1
    return counts


def other_plant_taxa(train: dict[str, Any], taxa: dict[str, dict[str, str]]) -> str:
    """The taxa `other_plant` was trained on, by scientific name."""
    category = next(c for c in train["categories"] if c["name"] == OTHER_CATEGORY)
    names = sorted(taxa[code]["name"] if code in taxa else code
                   for code in category["eppo_codes"])
    return ", ".join(f"*{name}*" for name in names)


def handoff_table(classifier_eval: Path, thresholds: list[float]) -> str:
    """Per species-probability threshold: the share of plants (not other_plant) the model
    names itself, and how often it is right on those, on the val crops."""
    rows = [r for r in read_csv(classifier_eval / "predictions.csv")
            if r["predicted"] != OTHER_CATEGORY]
    if not rows:
        return "(No classifier evaluation found.)"
    lines = ["| Threshold | Plants named by the model | Of those, species right |",
             "|---|---|---|"]
    for threshold in thresholds:
        kept = [r for r in rows if float(r["probability"]) >= threshold]
        right = sum(r["true"] == r["predicted"] for r in kept)
        lines.append(f"| {threshold} | {len(kept) / len(rows):.1%} | "
                     f"{right / len(kept):.1%} |" if kept else f"| {threshold} | 0% | - |")
    return "\n".join(lines)


def results_table(twostage_eval: Path, datasets: dict[str, Any]) -> str:
    """Both stages on the val split, per dataset (by_dataset.csv of predict_twostage.py)."""
    rows = [r for r in read_csv(twostage_eval / "by_dataset.csv") if r["dataset"] != "all"]
    if not rows:
        return "(No two-stage evaluation found.)"
    lines = ["| Dataset | Val images | Labelled plants | Plants found | Species right (of found) |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {datasets[r['dataset']]['name']} | {int(r['images']):,} | "
                     f"{int(r['plants']):,} | {share(r['found'])} | "
                     f"{share(r['species_right'])} |")
    return "\n".join(lines)


def classifier_table(classifier_eval: Path, datasets: dict[str, Any]) -> str:
    """The classifier alone on the val crops, per dataset (accuracy_by_dataset.csv)."""
    rows = read_csv(classifier_eval / "accuracy_by_dataset.csv")
    if not rows:
        return "(No classifier evaluation found.)"
    lines = ["| Dataset | Val plants | Species right | Balanced accuracy |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {datasets[r['dataset']]['name']} | {int(r['crops']):,} | "
                     f"{share(r['accuracy'])} | {share(r['balanced_accuracy'])} |")
    return "\n".join(lines)


def partial_section(classifier_eval: Path, datasets: dict[str, Any],
                    taxa: dict[str, dict[str, str]], trained: bool) -> str:
    """How often the classifier's answer fits plants labelled without their species
    (partial_labels.csv of evaluate_classifier.py); nothing when it has no such plants.
    `trained`: whether the classifier learned from them (data.partial_labels)."""
    path = classifier_eval / "partial_labels.csv"
    if not path.is_file():
        return ""
    learned = ("The classifier learned from these too, as \"one of the classes the label "
               "allows\"." if trained else "The classifier did not learn from these.")
    intro = ("Some plants are labelled without their species: only with their genus or "
             "family, or with the classes they can or can't be (PhenoBench's weeds: anything "
             f"but sugar beet). {learned} They can't be scored as right or wrong, but the "
             "classifier's answer can fit the label or not:")
    columns = ["Dataset", "Label", "Classes it allows", "Val plants", "Answer fits",
               "Most often called"]
    lines = ["", intro, "", f"| {' | '.join(columns)} |", "|---" * len(columns) + "|"]
    for r in read_csv(path):
        taxon = taxa.get(r["eppo"], {})
        label = (f"*{taxon['name']}*" if taxon.get("rank") in ("genus", "family", "order")
                 else "species unknown")
        lines.append(f"| {datasets[r['dataset']]['name']} | {label} | {r['allowed']} | "
                     f"{int(r['crops']):,} | {share(r['label_fit'])} | "
                     f"{r['most_predicted']} |")
    return "\n".join(lines) + "\n"


def dataset_table(train: dict[str, Any]) -> str:
    """One row per training dataset: source, licence, paper, resolution, what it trains."""
    by_dataset = defaultdict(list)
    for image in train["images"]:
        by_dataset[image.get("dataset")].append(image)
    lines = ["| Dataset | Licence | Ground resolution | Trains | Training images | Paper |",
             "|---|---|---|---|---|---|"]
    for key, dataset in train["info"]["datasets"].items():
        images = by_dataset[key]
        mm = [i["mm_per_px"] for i in images if i.get("mm_per_px")]
        estimated = any(i.get("mm_per_px_estimated") for i in images)
        resolution = (f"{median(mm):.2f} mm/px" + (" (estimated)" if estimated else "")
                      if mm else "unknown")
        box_only = all(i.get(BOX_ONLY) for i in images)
        trains = "classifier (boxes only)" if box_only else "detector and classifier"
        lines.append(f"| [{dataset['name']}]({dataset['source'].split(' ')[0]}) | "
                     f"{dataset['licence'].split('; see ')[0]} | {resolution} | {trains} | "
                     f"{len(images):,} | {PAPERS.get(key, '-')} |")
    return "\n".join(lines)


def split_rows(data_dirs: dict[str, Path]) -> dict[str, dict[str, str]]:
    """The images of the stages' datasets ({stage: folder with <split>.json}), by file name:
    their dataset, path inside the dataset's image folder, group, and the split each stage
    put them in (empty when the stage didn't use the image). No sha256 yet."""
    rows = {}
    for stage, folder in data_dirs.items():
        for split in SPLITS:
            coco = json.loads((folder / f"{split}.json").read_text(encoding="utf-8"))
            image_dirs = {name: d.get("image_dir", "").rstrip("/")
                          for name, d in coco["info"]["datasets"].items()}
            for image in coco["images"]:
                name = image["file_name"]
                row = rows.setdefault(name, {
                    "dataset": image["dataset"],
                    "image": name.removeprefix(image_dirs.get(image["dataset"], "") + "/"),
                    "sha256": "", "group": image.get("group", ""),
                    **{f"{s}_split": "" for s in data_dirs}})
                row[f"{stage}_split"] = split
    return rows


def write_splits(path: Path, data_dirs: dict[str, Path]) -> list[dict[str, str]]:
    """splits.csv: `split_rows` with the sha256 of each image file, sorted by dataset and
    image. Its rows."""
    rows = split_rows(data_dirs)
    missing = 0
    for file_name, row in tqdm(rows.items(), desc="sha256 of the images", unit="image"):
        image = resolve(file_name)
        if not image.is_file():
            missing += 1
            continue
        with open(image, "rb") as f:
            row["sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    if missing:
        print(f"{missing} images not found, left without sha256 in {path.name}")
    ordered = sorted(rows.values(), key=lambda r: (r["dataset"], r["image"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)
    return ordered


def split_table(rows: list[dict[str, str]]) -> str:
    """Images per split, for each stage of splits.csv."""
    stages = [key.removesuffix("_split") for key in rows[0] if key.endswith("_split")]
    lines = ["| Split | " + " | ".join(f"Images ({stage})" for stage in stages) + " |",
             "|---" * (len(stages) + 1) + "|"]
    for split in SPLITS:
        counts = [sum(r[f"{stage}_split"] == split for r in rows) for stage in stages]
        lines.append(f"| {split} | " + " | ".join(f"{n:,}" for n in counts) + " |")
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file; none (and a note) when it doesn't exist."""
    if not path.is_file():
        print(f"Not found, left out of the README: {path}")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def share(value: str) -> str:
    return f"{float(value):.1%}" if value else "-"
