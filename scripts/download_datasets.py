"""Download the datasets listed in configs/download.yaml, robustly over slow, unstable
connections: several connections per file, progress kept across interruptions and restarts,
stalled connections reopened, sizes and checksums verified (see weedseg.data.download).

Each dataset config's `download` section lists its files; archives go to
<raw_dir>/.archives/ and are unpacked into <raw_dir>. A file that brings nothing new for
`transfer.give_up_after_min` minutes is set aside so the others get their turn; the
unfinished files are listed at the end. Safe to stop and run again at any time.
"""

from weedseg.config import load_config
from weedseg.data.download import DownloadError, TransferSettings, fetch_all
from weedseg.paths import resolve

CONFIG = "configs/download.yaml"


def main(config_path: str = CONFIG) -> None:
    config = load_config(config_path)
    settings = TransferSettings.from_config(config["transfer"])
    datasets = []
    for dataset_config in config["datasets"]:
        dataset = load_config(dataset_config)
        datasets.append((dataset_config, dataset["download"], resolve(dataset["raw_dir"])))
    try:
        fetch_all(datasets, settings)
    except DownloadError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
