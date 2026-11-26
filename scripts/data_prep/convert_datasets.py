"""Convert every downloaded dataset in one go. Settings: configs/convert.yaml (the datasets,
each with its converter script, and the settings they must share).

Re-converts every listed dataset each time, so all of them follow the current settings. A
dataset whose raw_dir is missing, or holds only the download archives, is skipped with a
note. Before converting, it checks that the datasets agree on `same_settings` (e.g.
min_plant_size_mm), as scripts/data_prep/combine_datasets.py needs. A dataset that fails is
reported and the next one starts; the failures are listed at the end.

Each converter script converts one dataset with `convert(config_path)` and can still be run
on its own.
"""

import sys
import time
import traceback

from weedseg.config import load_config
from weedseg.data.convert_all import has_raw_data, setting_mismatches
from weedseg.paths import import_script, resolve

CONFIG = "configs/convert.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    datasets: dict[str, str] = config["datasets"]  # dataset config -> converter script
    configs = {path: load_config(path) for path in datasets}
    available = [path for path in datasets if has_raw_data(resolve(configs[path]["raw_dir"]))]
    problems = setting_mismatches({path: configs[path]["convert"] for path in available},
                                  config["same_settings"])
    if problems:
        raise SystemExit("The datasets' convert settings disagree, nothing converted:\n  "
                         + "\n  ".join(problems))

    done, skipped, failed = [], [], []
    for path, script in datasets.items():
        if path not in available:
            print(f"===== {path}: no data in {configs[path]['raw_dir']}, skipped")
            skipped.append(path)
            continue
        print(f"===== {path} ({script})")
        start = time.perf_counter()
        try:
            import_script(script).convert(path)
        except SystemExit as error:  # how the converters stop on missing or unusable input
            print(f"FAILED: {error}")
            failed.append(path)
        except Exception:  # noqa: BLE001 (any error ends that dataset only; printed here)
            sys.stdout.flush()  # so the error shows under its dataset's header
            traceback.print_exc()
            failed.append(path)
        else:
            print(f"----- {path} done in {(time.perf_counter() - start) / 60:.1f} min")
            done.append(path)

    print(f"\n{len(done)} converted, {len(skipped)} skipped (not downloaded), "
          f"{len(failed)} failed")
    for label, paths in (("skipped", skipped), ("failed", failed)):
        for path in paths:
            print(f"  {label}: {path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
