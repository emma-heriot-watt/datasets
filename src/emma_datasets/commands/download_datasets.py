import logging
from pathlib import Path
from typing import Optional, cast

from rich_click import typer

from emma_datasets.common import Downloader, Settings
from emma_datasets.common.downloader import DataDict
from emma_datasets.datamodels import DatasetName
from emma_datasets.io import read_csv


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    short_help="Download files for processing.",
)

logger = logging.getLogger(__name__)

settings = Settings()

DEFAULT_CSV_PATH = settings.paths.constants.joinpath("dataset_downloads.csv")


@app.command()
def download_datasets(
    datasets: Optional[list[DatasetName]] = typer.Argument(  # noqa: WPS404
        None,
        case_sensitive=False,
        help="Specify which datasets to download. Download all if none specified.",
    ),
    csv_file_path: Path = typer.Option(  # noqa: WPS404
        DEFAULT_CSV_PATH,
        help="Location of the CSV file which contians all the download locations.",
    ),
    output_dir: Path = typer.Option(  # noqa: WPS404
        settings.paths.datasets, help="Output directory for the files"
    ),
    max_concurrent_downloads: Optional[int] = typer.Option(  # noqa: WPS404
        None,
        help="Number of threads to use for parallel processing. This default to `min(32, os.cpu_count() + 4).",
    ),
) -> None:
    """Download the dataset files from the CSV file.

    If none are specified, download all of them.
    """
    if not datasets:
        datasets = list(DatasetName)

    downloader = Downloader()
    data_dicts = read_csv(csv_file_path)
    filtered_data_dicts = [
        cast(DataDict, file_dict)
        for file_dict in data_dicts
        if DatasetName[file_dict["dataset"]] in datasets
    ]

    downloader.download(filtered_data_dicts, output_dir, max_concurrent_downloads)


if __name__ == "__main__":
    app()
