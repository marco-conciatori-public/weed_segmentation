"""Deployment package of the plant detector: everything needed to run it outside this repo.

<output_dir>/<name>/ gets the weights as safetensors (plain tensors, no pickled code),
model.json (the settings plant_detector.py needs), plant_detector.py and requirements.txt as
they are, splits.csv (every image of the training data and the split it was in), and
README.md, made from README_template.md with the training data and the validation numbers of
the run. See docs/detector.md#deployment-package.
"""

import csv
import hashlib
import json
import platform
import re
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median
from string import Template
from typing import Any

import torch
import torchvision
from safetensors.torch import save_file
from tqdm import tqdm

from weedseg.config import load_config
from weedseg.data.combine import PLANT_CATEGORY, PLANT_ID, SPLITS
from weedseg.detector.model import ARCHITECTURE
from weedseg.detector.run import checkpoint_path
from weedseg.eppo import load_taxa
from weedseg.paths import resolve

HERE = Path(__file__).resolve().parent
COPIED = ("plant_detector.py", "requirements.txt")
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
    "lincolnbeet": "Darbyshire, Salazar-Gomez et al., \"Towards practical object detection for "
                   "weed spraying in precision agriculture\", Frontiers in Plant Science 2023",
    "imageweeds": "Rai et al., \"ImageWeeds: an image dataset consisting of weeds in multiple "
                  "formats to advance computer vision algorithms for real-time weed "
                  "identification and spot spraying application\", Data in Brief 2023",
    "vcd": "Lac et al., \"An annotated image dataset of vegetable crops at an early stage of "
           "growth for proximal sensing applications\", Data in Brief 2022",
}
# A dataset licence that rules out commercial use.
NON_COMMERCIAL = re.compile(r"non-commercial|\bNC\b", re.IGNORECASE)
# Species shown in the README's table of plants found: those with at least this many val plants.
MIN_SPECIES_PLANTS = 20


def export_package(config: dict[str, Any]) -> Path:
    """Write the package that `config` (configs/export_detector.yaml) describes; its folder."""
    detector_config = load_config(config["detector"])
    detector_file = checkpoint_path(detector_config, config["checkpoint"])
    checkpoint = torch.load(detector_file, map_location="cpu", weights_only=False)
    if [c["name"] for c in checkpoint["categories"]] != [PLANT_CATEGORY]:
        raise SystemExit(f"{detector_file} is not a plant detector (one class, plant)")

    created = date.today().isoformat()
    out_dir = resolve(config["output_dir"]) / config["name"].format(date=created)
    if out_dir.exists():
        if not config["overwrite"]:
            raise SystemExit(f"{out_dir} exists; set overwrite: true to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    data_dir = resolve(checkpoint["config"]["data"]["dir"])
    train = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    settings = checkpoint["config"]["model"]
    info = {
        "name": out_dir.name,
        "created": created,
        "kind": "plant detector (boxes)",
        "detector": {
            "architecture": f"{ARCHITECTURE}, one class (plant)",
            # several sizes are a training augmentation; evaluation uses the last
            "min_size": settings["min_size"][-1],
            "max_size": settings["max_size"],
            "anchor_sizes": settings["anchor_sizes"],
            "detections_per_image": settings["detections_per_image"],
            "score_threshold": settings["score_threshold"],
            "run": detector_file.parent.name, "epoch": checkpoint["epoch"],
        },
        "min_plant_score": config["min_plant_score"],
        "min_plant_size_mm": train["info"]["min_plant_size_mm"],
        "training_data": training_data(train),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__},
    }
    (out_dir / "model.json").write_text(json.dumps(info, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    save_file({k: v.contiguous() for k, v in checkpoint["model"].items()},
              out_dir / "detector.safetensors", metadata={"run": info["detector"]["run"]})
    for name in COPIED:
        shutil.copyfile(HERE / name, out_dir / name)
    splits = write_splits(out_dir / "splits.csv", data_dir)

    # as evaluate_detector.py names it (weedseg.detector.run.eval_dir), next to the checkpoint
    evaluation = (detector_file.parent / "eval"
                  / f"{detector_config['evaluate']['split']}_{detector_file.stem}")
    taxa = load_taxa(resolve(config["eppo_taxa"]))
    readme = readme_text(info, train, evaluation, taxa, splits)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    size = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"Package ({size:.0f} MB): {out_dir}")
    return out_dir


def training_data(train: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The datasets of the training data (its train.json): name, source and licence."""
    return {key: {"name": d.get("name", key), "source": d["source"],
                  "licence": d["licence"].split("; see ")[0]}
            for key, d in train["info"]["datasets"].items()}


def readme_text(info: dict[str, Any], train: dict[str, Any], evaluation: Path,
                taxa: dict[str, dict[str, str]], splits: list[dict[str, str]]) -> str:
    """README_template.md filled in; `train`: the training data; `evaluation`: the folder of
    evaluate_detector.py's outputs; `splits`: the rows of splits.csv."""
    datasets = info["training_data"]
    values = {
        "name": info["name"], "created": info["created"],
        "run": info["detector"]["run"], "epoch": info["detector"]["epoch"],
        "min_size": info["detector"]["min_size"], "max_size": info["detector"]["max_size"],
        "min_plant_size_mm": info["min_plant_size_mm"],
        "min_plant_score": info["min_plant_score"],
        **info["versions"],
        "results_table": results_table(evaluation, datasets),
        "species_table": species_table(evaluation, taxa),
        "dataset_table": dataset_table(train, datasets),
        "split_table": split_table(splits),
        "licence_text": licence_text(datasets),
    }
    template = Template((HERE / "README_template.md").read_text(encoding="utf-8"))
    return template.substitute(values)


def results_table(evaluation: Path, datasets: dict[str, Any]) -> str:
    """The detector on the val split, per dataset (by_dataset.csv of evaluate_detector.py).
    The first four columns are those the NeutraWeed platform reads."""
    rows = [r for r in read_csv(evaluation / "by_dataset.csv") if r["dataset"] != "all"]
    if not rows:
        return "(No evaluation found.)"
    lines = [("| Dataset | Val images | Labelled plants | Plants found | "
              "Detections on no labelled plant | Box AP50 |"), "|---|---|---|---|---|---|"]
    for r in rows:
        name = datasets.get(r["dataset"], {}).get("name", r["dataset"])
        lines.append(f"| {name} | {int(r['images']):,} | {int(r['plants']):,} | "
                     f"{share(r['found'])} | {int(r['detections_on_no_plant']):,} | "
                     f"{float(r['AP50']):.3f} |")
    return "\n".join(lines)


def species_table(evaluation: Path, taxa: dict[str, dict[str, str]]) -> str:
    """Plants found per species of their label (by_species.csv), for the species with at least
    MIN_SPECIES_PLANTS val plants."""
    rows = read_csv(evaluation / "by_species.csv")
    if not rows:
        return "(No evaluation found.)"
    shown = [r for r in rows if int(r["plants"]) >= MIN_SPECIES_PLANTS]
    lines = ["| Code | Label | Val plants | Plants found |", "|---|---|---|---|"] if shown else []
    for r in shown:
        name = r["name"] or r["eppo"]
        rank = taxa.get(r["eppo"], {}).get("rank")
        label = f"*{name}*" if rank in ("species", "genus") else name
        lines.append(f"| `{r['eppo']}` | {label} | {int(r['plants']):,} | {share(r['found'])} |")
    rest = len(rows) - len(shown)
    if rest:
        what = f"{rest} {'more ' * bool(shown)}label{'s have' if rest > 1 else ' has'}"
        lines.append("\n" * bool(shown) + f"{what} fewer than {MIN_SPECIES_PLANTS} val plants"
                     f"{' each' * (rest > 1)} (see by_species.csv of the evaluation).")
    return "\n".join(lines)


def dataset_table(train: dict[str, Any], datasets: dict[str, dict[str, Any]]) -> str:
    """One row per training dataset: source, licence, resolution, training images and
    plants, paper."""
    images = defaultdict(list)
    for image in train["images"]:
        images[image.get("dataset")].append(image)
    dataset_of = {image["id"]: image.get("dataset") for image in train["images"]}
    plants = Counter(dataset_of[a["image_id"]] for a in train["annotations"]
                     if a["category_id"] == PLANT_ID)
    lines = [("| Dataset | Licence | Ground resolution | Training images | Training plants | "
              "Paper |"), "|---|---|---|---|---|---|"]
    for key, dataset in datasets.items():
        mm = [i["mm_per_px"] for i in images[key] if i.get("mm_per_px")]
        estimated = any(i.get("mm_per_px_estimated") for i in images[key])
        resolution = (f"{median(mm):.2f} mm/px" + (" (estimated)" if estimated else "")
                      if mm else "unknown")
        lines.append(f"| [{dataset['name']}]({dataset['source'].split(' ')[0]}) | "
                     f"{dataset['licence']} | {resolution} | {len(images[key]):,} | "
                     f"{plants[key]:,} | {PAPERS.get(key, '-')} |")
    return "\n".join(lines)


def licence_text(datasets: dict[str, dict[str, Any]]) -> str:
    """What the training datasets' licences allow, for the README."""
    restricted = [d["name"] for d in datasets.values() if NON_COMMERCIAL.search(d["licence"])]
    if not restricted:
        return ("No training dataset has a non-commercial licence (see the table above); "
                "check each one's terms before commercial use, and cite the datasets (papers "
                "above) when you present results.")
    return ("The model was trained on datasets with non-commercial licences ("
            f"{', '.join(restricted)}; see the table above). So **use it only for "
            "non-commercial research and demonstration**, and cite the datasets (papers above) "
            "when you present results. Commercial use would need a model retrained without "
            "those datasets.")


def split_rows(data_dir: Path) -> dict[str, dict[str, str]]:
    """The images of the training data (data_dir/<split>.json), by file name: their dataset,
    path inside the dataset's image folder, group and split. No sha256 yet."""
    rows = {}
    for split in SPLITS:
        coco = json.loads((data_dir / f"{split}.json").read_text(encoding="utf-8"))
        image_dirs = {name: d.get("image_dir", "").rstrip("/")
                      for name, d in coco["info"]["datasets"].items()}
        for image in coco["images"]:
            name = image["file_name"]
            rows[name] = {
                "dataset": image["dataset"],
                "image": name.removeprefix(image_dirs.get(image["dataset"], "") + "/"),
                "sha256": "", "group": image.get("group", ""),
                # the platform reads this column name (the two-stage export's, per stage)
                "detector_split": split}
    return rows


def write_splits(path: Path, data_dir: Path) -> list[dict[str, str]]:
    """splits.csv: `split_rows` with the sha256 of each image file, sorted by dataset and
    image. Its rows."""
    rows = split_rows(data_dir)
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
    """Images per split and dataset, from splits.csv."""
    counts = Counter((r["dataset"], r["detector_split"]) for r in rows)
    datasets = sorted({r["dataset"] for r in rows})
    lines = ["| Dataset | " + " | ".join(SPLITS) + " |", "|---" * (len(SPLITS) + 1) + "|"]
    for dataset in datasets:
        lines.append(f"| {dataset} | "
                     + " | ".join(f"{counts[dataset, s]:,}" for s in SPLITS) + " |")
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
