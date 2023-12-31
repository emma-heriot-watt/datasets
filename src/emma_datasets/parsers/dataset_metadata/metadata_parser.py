import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel
from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName, DatasetSplit
from emma_datasets.io.paths import get_paths_from_dir


T = TypeVar("T", bound=BaseModel)

DataPathTuple = tuple[Path, Optional[DatasetSplit]]


class DatasetMetadataParser(ABC, Generic[T]):
    """Convert the metadata from raw dataset instances into the standardized `DatasetMetadata`.

    This class facilitates converting the metadata per instnace

    Subclasses should provide the class variables for `metadata_model` and `dataset_name`.
    """

    metadata_model: type[T]
    dataset_name: DatasetName
    file_ext: str = "json"
    feature_ext: str = "pt"

    def __init__(self, data_paths: list[DataPathTuple], progress: Progress) -> None:
        self.data_paths = self._get_all_file_paths(data_paths)

        self.task_id = progress.add_task(
            description=f"Structuring metadata from [u]{self.dataset_name.value}[/]",
            start=False,
            visible=False,
            total=0,
            comment="",
        )

    @abstractmethod
    def convert_to_dataset_metadata(self, metadata: T) -> DatasetMetadata:
        """Convert a single instance of metadata model to the common DatasetMetadata."""
        raise NotImplementedError()

    def get_metadata(self, progress: Progress) -> Iterator[T]:
        """Get all the raw metadata for this dataset."""
        structured_data_iterators: list[Iterator[T]] = []

        for path, dataset_split in self.data_paths:
            progress.update(
                self.task_id, visible=True, comment=f"Reading data from [u]{path.parts[-2:]}[/]"
            )
            raw_data = self._read(path)

            structured_data = self._structure_raw_metadata(raw_data, dataset_split, progress)
            structured_data_iterators.append(structured_data)

        return itertools.chain.from_iterable(structured_data_iterators)

    def _structure_raw_metadata(
        self,
        raw_metadata: list[dict[str, Any]],
        dataset_split: Optional[DatasetSplit],
        progress: Progress,
    ) -> Iterator[T]:
        """Structure raw metadata into a Pydantic model.

        This uses the class variable `metadata_model`.
        """
        raw_data = ({**metadata, "dataset_split": dataset_split} for metadata in raw_metadata)

        current_task_total = progress.tasks[self.task_id].total
        progress.update(
            self.task_id,
            visible=True,
            total=current_task_total + len(raw_metadata)
            if current_task_total is not None
            else None,
            comment="Structuring raw metadata",
        )
        progress.start_task(self.task_id)

        for raw_instance in raw_data:
            progress.advance(self.task_id)
            yield self.metadata_model.parse_obj(raw_instance)

        progress.update(self.task_id, comment="Done!")

    @abstractmethod
    def _read(self, path: Path) -> Any:
        """Read data from the given path."""
        raise NotImplementedError()

    def _get_all_file_paths(self, data_paths: list[DataPathTuple]) -> list[DataPathTuple]:
        """Convert any directories to file paths if they're given."""
        all_file_paths: list[DataPathTuple] = []

        for path, dataset_split in data_paths:
            # Verify the path is a directory before getting all the files within it.
            if not path.is_file() and path.is_dir():
                file_paths = (
                    path
                    for path in get_paths_from_dir(path)
                    if path.suffix.endswith(self.file_ext)
                )
                new_paths = zip(file_paths, itertools.cycle([dataset_split]))
                all_file_paths.extend(list(new_paths))
            else:
                # If it's not a directory, just let it go through.
                all_file_paths.append((path, dataset_split))

        return all_file_paths
