import itertools
import logging
import signal
import time
from collections.abc import Iterable, Iterator, Sized
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any, Optional, TypedDict, Union, cast
from urllib.parse import ParseResult, parse_qs, urlparse

import boto3
import requests
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TransferSpeedColumn,
)

from emma_datasets.common.progress import CustomTimeColumn
from emma_datasets.common.settings import Settings


settings = Settings()

overall_progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    MofNCompleteColumn(),
    CustomTimeColumn(),
)

overall_task = overall_progress.add_task("Download files")

job_progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="left"),
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.1f}%",
    DownloadColumn(),
    TransferSpeedColumn(),
    CustomTimeColumn(),
)


# Without this, boto print so many logs, it crashes the terminal.
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("nose").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

done_event = Event()

USER_AGENT = get_datasets_user_agent()


def handle_sigint(signum: int, frame: Any) -> None:
    """Handle SIGINT signal."""
    done_event.set()


TIME_STR = time.strftime("%Y%m%d-%H%M%S")
LOG_NAME = f"logs/downloader_{TIME_STR}.log"

HUGGINGFACE_URL_SCHEME = "hf"


@dataclass
class DownloadItem:
    """Class that represents an item to be downloaded."""

    output_path: Path
    url: str


@dataclass
class DownloadList:
    """Represents a generic list of items to be downloaded."""

    item_list: Iterator[DownloadItem]
    count: int


class DataDict(TypedDict):
    """A dataset to be downloaded."""

    url: str
    dataset: str


class Downloader:
    """Downloader files as fast as possible."""

    def __init__(self, log_file_path: str = LOG_NAME, chunk_size: int = 32768) -> None:
        self._log_file = Path(log_file_path)
        self._chunk_size = chunk_size

        self._create_log_file()

        signal.signal(signal.SIGINT, handle_sigint)

    def download(
        self, data_dicts: list[DataDict], output_dir: Path, max_workers: Optional[int] = None
    ) -> None:
        """Download a list of URLs to a specific directory from a set of datasets.

        The downloads happen in parallel across multiple workers, so set the `max_workers` argument
        if you do NOT want to use the maximum your machine can handle.
        """
        download_list = self._get_urls_from_data_dicts(data_dicts, output_dir)

        overall_progress.reset(overall_task, total=download_list.count)

        with self._display_progress():
            with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
                thread_pool.map(self.download_file, download_list.item_list)

    def download_file(self, data_item: DownloadItem) -> None:
        """Download the file to the specified directory.

        It will create the output directory if it does not exist.
        """
        url = data_item.url
        data_item.output_path.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        filename = filename.split("?")[0]

        dest_path = data_item.output_path.joinpath(filename)
        task_id = job_progress.add_task(
            "download",
            filename=filename,
            visible=False,
            start=False,
        )

        if url.startswith("s3"):
            download_func = self.download_from_s3
        else:
            download_func = self.download_from_url

        return download_func(task_id, url, dest_path)

    def download_from_s3(self, task_id: TaskID, url: str, path: Path) -> None:
        """Download file from a s3 bucket."""
        parsed_url = urlparse(url, allow_fragments=False)
        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")

        s3 = boto3.client("s3")

        is_directory = False
        if object_key.endswith("/"):
            # this URL points to a directory -- recursively download all the files
            raw_object_list = s3.list_objects(Bucket=bucket_name, Prefix=object_key)
            is_directory = True
            object_list = [raw_object["Key"] for raw_object in raw_object_list["Contents"]]
        else:
            object_list = [object_key]

        for object_id in object_list:  # noqa: WPS426
            file_size: int = s3.get_object_attributes(
                Bucket=bucket_name,
                Key=object_id,
                ObjectAttributes=["ObjectSize"],
            )["ObjectSize"]
            if is_directory:
                filename = path.joinpath(Path(object_id).name)
            else:
                filename = path

            job_progress.update(task_id, total=file_size)

            if not filename.exists() or filename.stat().st_size < file_size:
                job_progress.start_task(task_id)
                job_progress.update(task_id, visible=True)

                s3.download_file(
                    Bucket=bucket_name,
                    Key=object_id,
                    Filename=filename.as_posix(),
                    Callback=lambda x: job_progress.update(task_id, advance=x),
                )

            job_progress.reset(task_id)

        self._complete_download(task_id, path.name)

    def download_from_url(self, task_id: TaskID, url: str, path: Path) -> None:
        """Copy data from a url to a local file."""
        if done_event.is_set():
            self._handle_abort_event(task_id, path.name)
            return

        try:
            with requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=True,
                timeout=5,
                stream=True,
            ) as response:
                content_length = self._get_content_length_from_response(response)
                content_type = self._get_content_type_from_response(response)
                if not path.suffix and content_type == "image/jpeg":
                    # if we're missing the extension for a JPEG file we add it here
                    path = path.with_suffix(".jpg")
                job_progress.update(task_id, total=content_length)

                if not path.exists() or path.stat().st_size < content_length:
                    self._store_data(response, path, task_id)

            self._complete_download(task_id, path.name)

        except requests.exceptions.Timeout:
            self._log_file.write_text(
                f"[Timeout]: Unable to download data from URL because timed out after 5 seconds: {url}"
            )
        except requests.exceptions.TooManyRedirects:
            self._log_file.write_text(
                f"[TooManyRedirects]: Unable to download data from URL: {url}"
            )
        except requests.exceptions.RequestException:
            self._log_file.write_text(
                f"[RequestException]: Unable to download data from URL: {url}"
            )

    def _store_data(self, response: requests.Response, path: Path, task_id: TaskID) -> None:
        """Stores the data returned with an HTTP response."""
        with open(path, "wb") as dest_file:
            job_progress.start_task(task_id)

            for data in response.iter_content(self._chunk_size):
                if done_event.is_set():
                    self._handle_abort_event(task_id, path.name)
                    return

                dest_file.write(data)
                job_progress.update(task_id, advance=len(data), visible=True)

    def _get_urls_from_hf_dataset(
        self, parsed_url: ParseResult, dataset_name: str, output_dir: Path
    ) -> tuple[Iterator[DownloadItem], int]:
        """Given an Huggingface dataset URL, derive a sequence of URLs that can be downloaded.

        We define an Huggingface dataset by adopting a specific URL format that is structured as
        follows:

        `hf://<dataset_identifier>?key=<key_to_access_url>&split=<split_id>`

        where:
        - `<dataset_identifier>`: Huggingface dataset identifier
        - `<key_to_access_url>`: field of the dataset containing the URL
        - `<split_id>`: reference split of the dataset (depending on the release)

        Args:
            parsed_url (ParseResult): Reference URL for Huggingface dataset
            dataset_name (str): Huggingface dataset idenfier
            output_dir (Path): reference directory for the dataset

        Returns:
            tuple[Iterator[DownloadItem], int]: iterator of downloadable items with a possible
            count. The count is available only if the dataset is not Iterable.
        """
        query_params = parse_qs(parsed_url.query)
        item_key = query_params["key"][0]
        split = query_params["split"][0]
        dataset = load_dataset(dataset_name, split=split)

        data_iterator = (
            DownloadItem(
                output_path=output_dir.joinpath(split),
                url=data_item[item_key],
            )
            for data_item in dataset
        )

        data_count = len(dataset) if isinstance(dataset, Sized) else 0

        return data_iterator, data_count

    def _get_urls_from_data_dicts(
        self, data_dicts: list[DataDict], output_dir: Path
    ) -> DownloadList:
        """Returns a list of URLs that can be downloaded in parallel.

        Most of the datasets have specific archives or files that contain the dataset annotations.
        The only exception is Huggingface for which we have a datasets object.
        """
        data_items: list[Iterable[DownloadItem]] = []
        count = 0

        for dict_item in data_dicts:
            url = dict_item["url"]
            dataset_name = dict_item["dataset"]

            parsed_url = urlparse(url)
            output_path = output_dir.joinpath(dataset_name)

            if parsed_url.scheme == HUGGINGFACE_URL_SCHEME:
                data_iterator, data_count = self._get_urls_from_hf_dataset(
                    parsed_url, dataset_name, output_path
                )
                data_items.append(data_iterator)

                count += data_count

            else:
                data_items.append([DownloadItem(output_path=output_path, url=url)])
                count += 1

        return DownloadList(item_list=itertools.chain.from_iterable(data_items), count=count)

    def _complete_download(self, task_id: TaskID, file_name: str) -> None:
        """Complete the download and update the progress bars."""
        job_progress.update(task_id, visible=False)
        overall_progress.console.log(f"Downloaded {file_name}")

        overall_progress.advance(overall_task)

        self._update_log_file(file_name)

    def _handle_abort_event(self, task_id: TaskID, filename: str) -> None:
        """Stop the task and prevent any new ones from starting."""
        self._update_log_file(filename, "Aborted")
        job_progress.stop_task(task_id)

        if job_progress.tasks[task_id].completed < 1:
            job_progress.update(task_id, visible=False)

    def _create_log_file(self) -> None:
        """Create the log file using Rich Console API."""
        with open(self._log_file, "w") as log_file:
            console = Console(file=log_file)
            console.rule("Starting download")

    def _update_log_file(self, path: str, verb: str = "Downloaded") -> None:
        """Update the log file using Rich Console API."""
        with open(self._log_file, "a") as log_file:
            console = Console(file=log_file)
            console.log(f"{verb} {path}")

    def _get_content_length_from_response(self, response: requests.Response) -> Union[int, float]:
        """Get content length from response."""
        content_length = response.headers.get(
            "X-Dropbox-Content-Length", response.headers.get("Content-Length")
        )

        return int(content_length) if content_length is not None else float("inf")

    def _get_content_type_from_response(self, response: requests.Response) -> str:
        """Get content type from HTTP response."""
        return response.headers["Content-Type"]

    def _display_progress(self) -> Live:
        """Return a rich `Live` object to display the progress bars.

        This should be used as a context manager.
        """
        progress_group = Group(
            cast(RenderableType, overall_progress),
            Panel(cast(RenderableType, job_progress)),
        )

        return Live(progress_group)
