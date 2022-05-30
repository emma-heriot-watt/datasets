import logging
from pathlib import Path
from typing import Optional

import typer

from emma_datasets.common import Downloader, Settings
from emma_datasets.datamodels import DatasetName
from emma_datasets.io import read_csv


logger = logging.getLogger(__name__)

settings = Settings()

DEFAULT_CSV_PATH = settings.paths.constants.joinpath("dataset_downloads.csv")


def download_datasets(
    datasets: Optional[list[DatasetName]] = typer.Argument(  # noqa: WPS404
        None, case_sensitive=False, show_default=False
    ),
    csv_file_path: Path = DEFAULT_CSV_PATH,
    output_dir: Path = settings.paths.datasets,
    max_concurrent_downloads: int = 1,
) -> None:
    """Download the dataset files from the CSV file.

    If none are specified, download all of them.
    """
    if not datasets:
        logger.info("No datasets provided, therefore downloading all datasets...")
        datasets = list(DatasetName)

    downloader = Downloader()
    data_dicts = read_csv(csv_file_path)

    all_urls = []
    for file_dict in data_dicts:
        dataset, url = file_dict.values()
        if DatasetName[dataset] in datasets:
            all_urls.append(url)

    downloader.download(all_urls, output_dir.joinpath(dataset), max_concurrent_downloads)


if __name__ == "__main__":
    download_datasets()
