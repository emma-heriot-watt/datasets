from abc import ABC, abstractmethod
from typing import Any, Generic, Iterator, Optional, TypeVar

from pydantic import BaseModel
from rich.progress import Progress

from src.datamodels import DatasetMetadata, DatasetName, DatasetSplit


T = TypeVar("T", bound=BaseModel)


class DatasetMetadataParser(ABC, Generic[T]):
    """Parse dataset metadata and optionally account for splits.

    Subclasses should provide the class variables for `metadata_model` and `dataset_name`.
    """

    metadata_model: type[T]
    dataset_name: DatasetName

    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.task_id = progress.add_task(
            description=f"Structuring image metadata from [u]{self.dataset_name.value}[/]",
            start=False,
            visible=False,
            total=0,
        )

    @abstractmethod
    def get_metadata(self) -> Iterator[T]:
        """Get all the raw metadata for this dataset."""
        raise NotImplementedError()

    @abstractmethod
    def convert_to_dataset_metadata(self, metadata: T) -> DatasetMetadata:
        """Convert a single instance of metadata model to the common DatasetMetadata."""
        raise NotImplementedError()

    def structure_raw_metadata(
        self,
        raw_metadata: list[dict[str, Any]],
        dataset_split: Optional[DatasetSplit] = None,
    ) -> Iterator[T]:
        """Structure raw metadata into a Pydantic model.

        This uses the class variable `metadata_model`.
        """
        raw_data = ({**metadata, "dataset_split": dataset_split} for metadata in raw_metadata)

        self.progress.update(
            self.task_id,
            visible=True,
            total=self.progress._tasks[self.task_id].total + len(raw_metadata),  # noqa: WPS437
        )
        self.progress.start_task(self.task_id)

        for raw_instance in raw_data:
            parsed_model = self.metadata_model.parse_obj(raw_instance)

            self.progress.advance(self.task_id)
            yield parsed_model
