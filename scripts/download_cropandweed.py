"""Download and extract the CropAndWeed dataset. Settings: configs/cropandweed.yaml (download).

Safe to re-run: finished parts are skipped and an interrupted download resumes.
"""

from weedseg.config import load_config
from weedseg.data.download import download_and_extract_tar
from weedseg.paths import resolve

CONFIG = "configs/cropandweed.yaml"


def main() -> None:
    config = load_config(CONFIG)
    raw_dir = resolve(config["raw_dir"])
    settings = config["download"]
    for part in settings["parts"]:
        marker = raw_dir / ".downloaded" / part
        if settings["skip_existing"] and marker.exists():
            print(f"{part}: already downloaded, skipping")
            continue
        download_and_extract_tar(f"{settings['url_prefix']}{part}.tar", raw_dir, marker,
                                 settings["timeout_s"], settings["max_retries"],
                                 settings["keep_archives"])


if __name__ == "__main__":
    main()
