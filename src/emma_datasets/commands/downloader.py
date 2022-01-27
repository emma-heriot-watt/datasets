import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from rich.console import Console
from rich.progress import DownloadColumn, TaskID, TextColumn, TransferSpeedColumn

from emma_datasets.common import Settings
from emma_datasets.common.progress import CustomBarColumn, CustomProgress, CustomTimeColumn
from emma_datasets.datamodels import DatasetName
from emma_datasets.io import read_csv


settings = Settings()
settings.paths.create_dirs()

progress = CustomProgress(
    TextColumn("[white]{task.fields[dataset_name]}", justify="left"),
    TextColumn("[bold blue]{task.fields[filename]}", justify="left"),
    CustomBarColumn(),
    "[progress.percentage]{task.percentage:>3.1f}%",
    DownloadColumn(),
    TransferSpeedColumn(),
    CustomTimeColumn(),
)

done_event = Event()

# Without this, boto print so many logs, it crashes the terminal.
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("nose").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def handle_sigint(signum: int, frame: Any) -> None:
    """Handle SIGINT signal."""
    done_event.set()


signal.signal(signal.SIGINT, handle_sigint)

TIME_STR = time.strftime("%Y%m%d-%H%M%S")
LOG_NAME = f"downloader_{TIME_STR}.log"


class DatasetDownloader:
    """Download files for the datasets.

    After setup, call the `download_file()` method to download the dataset's file to the specified directory.

    There is currently no post-processing involved that extracts the content from the downloaded archives.
    """

    def __init__(self, log_name: str = LOG_NAME) -> None:
        self._log_file = Path(log_name)
        self._chunk_size = 32768

        self._create_log_file()

    def download_file(
        self,
        dataset_name: DatasetName,
        url: str,
        output_dir: Path,
        pool: ThreadPoolExecutor,
    ) -> None:
        """Download the file to the specified directory.

        It will create the output directory if it does not exist.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        dest_path = output_dir.joinpath(filename)
        task_id = progress.add_task(
            "download",
            dataset_name=dataset_name.value,
            filename=filename,
            visible=False,
            start=False,
        )

        if url.startswith("s3"):
            download_func = self._download_file_from_bucket
        else:
            download_func = self._download_file

        pool.submit(download_func, task_id, url, dest_path)

    def _download_file_from_bucket(self, task_id: TaskID, url: str, path: Path) -> None:
        """Download file from a s3 bucket."""
        s3_resource = boto3.resource(
            "s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED)
        )
        parsed_url = urlparse(url, allow_fragments=False)
        bucket_name = parsed_url.netloc
        key = parsed_url.path.lstrip("/")

        file_object = s3_resource.Object(bucket_name=bucket_name, key=key)
        bucket = s3_resource.Bucket(bucket_name)

        # This will break if the response doesn't contain content length
        progress.update(task_id, total=file_object.content_length)

        if not path.exists() or path.stat().st_size < file_object.content_length:
            progress.start_task(task_id)
            progress.update(task_id, visible=True)

            bucket.download_file(
                Key=key,
                Filename=path.as_posix(),
                Callback=lambda x: progress.update(task_id, advance=x),
            )

            progress.update(task_id, visible=False)
            progress.console.log(f"Downloaded {path.name}")
        self._update_log_file(path.name)

    def _download_file(self, task_id: TaskID, url: str, path: Path) -> None:
        """Copy data from a url to a local file."""
        response = urlopen(url)  # noqa: S310

        content_length = int(response.info()["Content-length"])

        # This will break if the response doesn't contain content length
        progress.update(task_id, total=content_length)

        if not path.exists() or path.stat().st_size < content_length:
            with open(path, "wb") as dest_file:
                progress.start_task(task_id)

                for data in iter(partial(response.read, self._chunk_size), b""):
                    dest_file.write(data)
                    progress.update(task_id, advance=len(data), visible=True)

                    if done_event.is_set():
                        self._update_log_file(path.name, "Aborted")  # noqa: WPS220
                        return  # noqa: WPS220

            progress.update(task_id, visible=False)
            progress.console.log(f"Downloaded {path.name}")
        self._update_log_file(path.name)

    def _create_log_file(self) -> None:
        with open(self._log_file, "w") as log_file:
            console = Console(file=log_file)
            console.rule("Starting download")

    def _update_log_file(self, path: str, verb: str = "Downloaded") -> None:
        with open(self._log_file, "a") as log_file:
            console = Console(file=log_file)
            console.log(f"{verb} {path}")


def download(csv_file_path: str, storage_root: str, max_workers: Optional[int] = None) -> None:
    """Download the dataset files from the CSV file of urls."""
    downloader = DatasetDownloader()
    data_dicts = read_csv(csv_file_path)

    with progress:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for file_dict in data_dicts:
                dataset, url = file_dict.values()
                downloader.download_file(
                    dataset_name=DatasetName[dataset],
                    url=url,
                    output_dir=Path(storage_root).joinpath(dataset),
                    pool=pool,
                )


if __name__ == "__main__":
    download("dataset_downloads.csv", "storage/datasets")
