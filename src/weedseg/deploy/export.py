"""Deployment package of the two-stage model: everything needed to run it outside this repo.

<output_dir>/<name>/ gets the weights of both stages as safetensors (plain tensors, no
pickled code), model.json (the settings weed_model.py needs, the species list and the
hand-off thresholds), weed_model.py and requirements.txt as they are, and README.md, made
from README_template.md with the species, training data and validation numbers of the runs.
See docs/twostage.md#deployment-package.
"""

import csv
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

from weedseg.classifier.crops import FILL_RGB
from weedseg.classifier.data import MEAN, STD
from weedseg.config import load_config
from weedseg.data.coco import BOX_ONLY, IGNORE_CATEGORY, OTHER_CATEGORY
from weedseg.eppo import NAME_SEPARATOR, load_taxa
from weedseg.maskrcnn.run import checkpoint_path
from weedseg.paths import resolve

HERE = Path(__file__).resolve().parent
COPIED = ("weed_model.py", "requirements.txt")
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
            "input_size": crops["input_size"], "margin": crops["margin"],
            "fill_rgb": list(FILL_RGB), "mean": MEAN.flatten().tolist(),
            "std": STD.flatten().tolist(), "batch_size": twostage["classifier"]["batch_size"],
            "run": classifier_file.parent.name, "epoch": classifier["epoch"],
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
    for name in COPIED:
        shutil.copyfile(HERE / name, out_dir / name)

    twostage_eval = (resolve(twostage["output_dir"])
                     / twostage["run_name"].format(detector=detector_file.parent.name,
                                                   classifier=classifier_file.parent.name)
                     / "eval" / twostage["evaluate"]["split"])
    # as evaluate_classifier.py names it (weedseg.maskrcnn.run.eval_dir), next to the checkpoint
    classifier_eval = (classifier_file.parent / "eval"
                       / f"{classifier_config['evaluate']['split']}_{classifier_file.stem}")
    readme = readme_text(info, train, classifier["classes"], taxa, twostage_eval,
                         classifier_eval, config["handoff_table"])
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"Package ({size:.0f} MB): {out_dir}")
    return out_dir


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
                thresholds: list[float]) -> str:
    """README_template.md filled in."""
    datasets = train["info"]["datasets"]
    values = {
        "name": info["name"], "created": info["created"],
        "detector_run": info["detector"]["run"], "detector_epoch": info["detector"]["epoch"],
        "classifier_run": info["classifier"]["run"],
        "classifier_epoch": info["classifier"]["epoch"],
        "backbone": info["classifier"]["backbone"],
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
        "dataset_table": dataset_table(train),
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


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file; none (and a note) when it doesn't exist."""
    if not path.is_file():
        print(f"Not found, left out of the README: {path}")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def share(value: str) -> str:
    return f"{float(value):.1%}" if value else "-"
