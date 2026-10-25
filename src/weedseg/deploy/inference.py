"""Run an exported two-stage package on any images (scripts/demo_twostage.py): the
plants drawn on each image and saved as JSON, with no labels and no evaluation."""

import importlib.util
import json
import webbrowser
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from weedseg.paths import resolve
from weedseg.visualize import draw_plants

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def image_files(images: str | list[str]) -> list[Path]:
    """The images named by `images`: a file, a folder (its images, not those of its
    subfolders) or a list of both. Their names must differ, as they name the results."""
    files = []
    for entry in [images] if isinstance(images, str) else images:
        path = resolve(entry)
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"No image or folder at {path}")
    clashes = [stem for stem, n in Counter(p.stem for p in files).items() if n > 1]
    if clashes:
        raise ValueError(f"Several images named {clashes}: their results would overwrite "
                         f"each other")
    return files


def load_package(folder: Path, device: str = "auto", amp: bool = True) -> Any:
    """The WeedModel of an exported package, run by the package's own weed_model.py, so the
    code is the one shipped with these weights even after weedseg changes."""
    spec = importlib.util.spec_from_file_location(f"weed_model_{folder.name}",
                                                  folder / "weed_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WeedModel(folder, device=None if device == "auto" else device, amp=amp)


def run_inference(config: dict[str, Any]) -> None:
    folder = resolve(config["model"])
    files = image_files(config["images"])
    if not files:
        raise SystemExit(f"No images in {config['images']}")
    start = perf_counter()
    model = load_package(folder, config["device"], config["amp"])
    print(f"model {folder.name} on {model.device} (loaded in {perf_counter() - start:.1f} s), "
          f"{len(files)} images")
    out_dir = resolve(config["output_dir"].format(model=folder.name))
    out_dir.mkdir(parents=True, exist_ok=True)
    draw = config["draw"]

    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Can't read the image {path}")
        start = perf_counter()
        plants = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                               min_plant_score=config["min_plant_score"])
        seconds = perf_counter() - start
        drawn_path = out_dir / f"{path.stem}.jpg"
        cv2.imwrite(str(drawn_path), draw_plants(image, plants, model.codes, draw["label"],
                                                 draw["mask_alpha"]))
        record = {"image": str(path), "width": image.shape[1], "height": image.shape[0],
                  "model": folder.name, "plants": model.to_json(plants)}
        (out_dir / f"{path.stem}.json").write_text(json.dumps(record, indent=1),
                                                   encoding="utf-8")
        named = Counter(p["species"] for p in plants if not p["send_to_zero_shot"])
        zero_shot = len(plants) - named.total()
        print(f"{path.name}: {len(plants)} plants in {seconds:.1f} s, {zero_shot} to zero-shot"
              + "".join(f", {code} {n}" for code, n in named.most_common()))
        if config["open_results"]:
            webbrowser.open(drawn_path.as_uri())
    print(f"results in {out_dir}")
