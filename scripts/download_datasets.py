"""Download the datasets listed in configs/download.yaml, robustly over slow, unstable
connections: several connections per file, progress kept across interruptions and restarts,
stalled connections reopened, sizes and checksums verified (see weedseg.data.download).

Each dataset config's `download` section lists its files; archives go to
<raw_dir>/.archives/ and are unpacked into <raw_dir>. Safe to stop and run again at any time.
"""

from weedseg.config import load_config
from weedseg.data.download import TransferSettings, fetch_files
from weedseg.paths import resolve

CONFIG = "configs/download.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    settings = TransferSettings.from_config(config["transfer"])
    for dataset_config in config["datasets"]:
        dataset = load_config(dataset_config)
        print(f"===== {dataset_config} -> {dataset['raw_dir']}")
        fetch_files(dataset["download"]["files"], resolve(dataset["raw_dir"]), settings,
                    dataset["download"].get("keep_archives", False))


if __name__ == "__main__":
    main()
