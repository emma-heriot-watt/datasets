import argparse
from pathlib import Path

from emma_datasets.common import Downloader, Settings
from emma_datasets.io import read_csv


settings = Settings()


def download_datasets(csv_file_path: Path, storage_root: Path, max_workers: int) -> None:
    """Download the dataset files from the CSV file of urls."""
    downloader = Downloader()
    data_dicts = read_csv(csv_file_path)

    all_urls = []
    for file_dict in data_dicts:
        dataset, url = file_dict.values()
        all_urls.append(url)

    downloader.download(all_urls, storage_root.joinpath(dataset), max_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv-file-path",
        help="Path to the CSV file of dataset files to download",
        type=Path,
        default=settings.paths.constants.joinpath("dataset_downloads.csv"),
    )
    parser.add_argument(
        "--output-dir",
        help="Path to the datasets directory",
        type=Path,
        default=settings.paths.datasets,
    )
    parser.add_argument(
        "--max-workers",
        help="Maximum number of concurrent downloads at any one time",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    download_datasets(args.csv_file_path, args.output_dir, args.max_workers)
