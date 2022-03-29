import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.client import HTTPResponse
from pathlib import Path
from threading import Event
from typing import Any, Optional, Union, cast
from urllib.parse import urlparse
from urllib.request import urlopen

import boto3
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

from emma_datasets.common.progress import CustomBarColumn, CustomProgress, CustomTimeColumn
from emma_datasets.common.settings import Settings


settings = Settings()

overall_progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    MofNCompleteColumn(),
    CustomTimeColumn(),
)

overall_task = overall_progress.add_task("Download files")

job_progress = CustomProgress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="left"),
    CustomBarColumn(),
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


def handle_sigint(signum: int, frame: Any) -> None:
    """Handle SIGINT signal."""
    done_event.set()


signal.signal(signal.SIGINT, handle_sigint)

TIME_STR = time.strftime("%Y%m%d-%H%M%S")
LOG_NAME = f"logs/downloader_{TIME_STR}.log"


class Downloader:
    """Downloader files as fast as possible."""

    def __init__(self, log_file_path: str = LOG_NAME, chunk_size: int = 32768) -> None:
        self._log_file = Path(log_file_path)
        self._chunk_size = chunk_size

        self._create_log_file()

    def download(
        self, urls: list[str], output_dir: Path, max_workers: Optional[int] = None
    ) -> None:
        """Download a list of URLS to a specific directory.

        The downloads happen in parallel across multiple workers, so set the `max_workers` argument
        if you do NOT want to use the maximum your machine can handle.
        """
        overall_progress.reset(overall_task, total=len(urls))

        with self._display_progress():
            with ThreadPoolExecutor(max_workers=max_workers) as thread_pool:
                for url in urls:
                    self.download_file(url=url, output_dir=output_dir, pool=thread_pool)

    def download_file(self, url: str, output_dir: Path, pool: ThreadPoolExecutor) -> None:
        """Download the file to the specified directory.

        It will create the output directory if it does not exist.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        filename = filename.split("?")[0]
        dest_path = output_dir.joinpath(filename)
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

        pool.submit(download_func, task_id, url, dest_path)

    def download_from_s3(self, task_id: TaskID, url: str, path: Path) -> None:
        """Download file from a s3 bucket."""
        parsed_url = urlparse(url, allow_fragments=False)
        bucket_name = parsed_url.netloc
        object_key = parsed_url.path.lstrip("/")

        s3 = boto3.resource("s3")
        file_object = s3.Object(bucket_name, object_key)

        job_progress.update(task_id, total=file_object.content_length)

        if not path.exists() or path.stat().st_size < file_object.content_length:
            job_progress.start_task(task_id)
            job_progress.update(task_id, visible=True)

            s3.download_file(
                Filename=path.as_posix(),
                Callback=lambda x: job_progress.update(task_id, advance=x),
            )

        self._complete_download(task_id, path.name)

    def download_from_url(self, task_id: TaskID, url: str, path: Path) -> None:
        """Copy data from a url to a local file."""
        if done_event.is_set():
            self._handle_abort_event(task_id, path.name)
            return

        response = urlopen(url)  # noqa: S310

        content_length = self._get_content_length_from_response(response)
        job_progress.update(task_id, total=content_length)

        if not path.exists() or path.stat().st_size < content_length:
            with open(path, "wb") as dest_file:
                job_progress.start_task(task_id)

                for data in iter(partial(response.read, self._chunk_size), b""):
                    if done_event.is_set():
                        self._handle_abort_event(task_id, path.name)  # noqa: WPS220
                        return  # noqa: WPS220

                    dest_file.write(data)
                    job_progress.update(task_id, advance=len(data), visible=True)

        self._complete_download(task_id, path.name)

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

    def _get_content_length_from_response(self, response: HTTPResponse) -> Union[int, float]:
        """Get content length from response."""
        if response.info()["Content-length"] is not None:
            return int(response.info()["Content-length"])

        if response.info()["X-Dropbox-Content-Length"] is not None:
            return int(response.info()["X-Dropbox-Content-Length"])

        return float("inf")

    def _display_progress(self) -> Live:
        """Return a rich `Live` object to display the progress bars.

        This should be used as a context manager.
        """
        progress_group = Group(
            cast(RenderableType, overall_progress),
            Panel(cast(RenderableType, job_progress)),
        )

        return Live(progress_group)
