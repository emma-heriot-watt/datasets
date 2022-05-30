import itertools
from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, Optional, TypeVar, Union, overload

from pydantic import BaseModel
from rich.progress import Progress

from emma_datasets.io import get_all_file_paths, read_json, write_json


Annotation = TypeVar("Annotation", bound=BaseModel)


class AnnotationExtractor(ABC, Generic[Annotation]):
    """Extract annotations from the raw dataset into multiple files for easier loading."""

    progress_bar_description = "Extracting annotations"
    file_ext: str = "json"

    def __init__(
        self,
        paths: Union[str, list[str], Path, list[Path]],
        output_dir: Union[str, Path],
        progress: Progress,
    ) -> None:
        self.task_id = progress.add_task(
            self.progress_bar_description,
            start=False,
            visible=True,
            total=float("inf"),
            comment="",
        )

        progress.update(self.task_id, comment="Getting paths to raw files")
        self.file_paths = [
            path for path in get_all_file_paths(paths) if path.suffix.endswith(self.file_ext)
        ]
        self.output_dir = Path(output_dir)

        progress.update(self.task_id, comment="Waiting for turn...")

    @overload
    def run(self, progress: Progress, pool: Pool) -> None:
        ...  # noqa: WPS428

    @overload
    def run(self, progress: Progress) -> None:
        ...  # noqa: WPS428

    def run(self, progress: Progress, pool: Optional[Pool] = None) -> None:
        """Run the splitter.

        Args:
            progress (Progress): Rich Progress Bar
            pool (Pool, optional): Pool for multiprocessing. Defaults to None.
        """
        progress.update(self.task_id, comment="Reading all raw data")
        raw_data = self._read()

        self._start_progress(progress)

        progress.update(self.task_id, comment="Processing data")
        if pool is not None:
            for _ in pool.imap_unordered(self.process_single_instance, raw_data):
                self._advance(progress)
        else:
            for raw_input in raw_data:
                self.process_single_instance(raw_input)
                self._advance(progress)

        self._end_progress(progress)

    def process_raw_file_return(self, raw_data: Any) -> Any:
        """Modify what is returned immediately after reading each file.

        See `_read()` method for how this is used.
        """
        return raw_data

    def postprocess_raw_data(self, raw_data: Any) -> Iterator[Any]:
        """Process all the raw data from all files.

        See `_read()` method for how this is used.
        """
        return raw_data

    @abstractmethod
    def convert(self, raw_feature: Any) -> Union[Annotation, Iterable[Annotation]]:
        """Convert a raw annotation into a Annotation."""
        raise NotImplementedError()

    @abstractmethod
    def process_single_instance(self, raw_feature: Any) -> None:
        """Process raw instance from the loaded data.

        Generally, take raw data, convert it, and then write it to a file.

        See other modules for how this is used.
        """
        raise NotImplementedError()

    def read(self, file_path: Path) -> Any:
        """Read the file using orjson.

        Overwrite if this is not good for your needs.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        return read_json(file_path)

    def _read(self) -> Iterator[Any]:
        """Read all files and return a single Iterator over all of them."""
        raw_data = itertools.chain.from_iterable(
            self.process_raw_file_return(self.read(file_path)) for file_path in self.file_paths
        )

        return self.postprocess_raw_data(raw_data)

    def _write(
        self,
        features: Union[Annotation, Iterable[Annotation]],
        filename: str,
        ext: str = "json",
    ) -> None:
        """Write the data to a JSON file using orjson."""
        filepath = self.output_dir.joinpath(f"{filename}.{ext}")

        features_dict = (
            [feature.dict(by_alias=True) for feature in features]
            if isinstance(features, Iterable) and not isinstance(features, BaseModel)
            else features.dict(by_alias=True)
        )

        write_json(filepath, features_dict)

    def _start_progress(self, progress: Progress) -> None:
        """Start the task on the progress bar."""
        progress.reset(self.task_id, start=True, visible=True)

    def _advance(self, progress: Progress) -> None:
        """Update the progress bar."""
        progress.advance(self.task_id)

    def _end_progress(self, progress: Progress) -> None:
        """Stop the progress bar and make sure to freeze the finished bar."""
        completed = int(progress._tasks[self.task_id].completed)  # noqa: WPS437
        progress.update(
            self.task_id, visible=True, total=completed, completed=completed, comment="Done!"
        )
        progress.stop_task(self.task_id)