from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Generic, Iterable, Optional, TypeVar, Union

from rich.progress import BarColumn, Progress, TaskID, TimeElapsedColumn

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from emma_datasets.common import Settings
from emma_datasets.common.progress import BatchesProcessedColumn, ProcessingSpeedColumn
from emma_datasets.datamodels import BaseInstance, DatasetName, DatasetSplit
from emma_datasets.db import DatasetDb
from emma_datasets.db.storage import StorageType, TorchStorage
from emma_datasets.parsers.instance_creators import DownstreamInstanceCreator


settings = Settings()

DatasetSplitSourceType = TypeVar(
    "DatasetSplitSourceType",
    Path,
    list[Path],
    list[dict[Any, Any]],
    Union[Dataset, IterableDataset],
)
InstanceModelType = TypeVar("InstanceModelType", bound=BaseInstance)


def create_downstream_rich_progress() -> Progress:
    """Create a Rich Progress tracker for the creator."""
    return Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        BatchesProcessedColumn(),
        TimeElapsedColumn(),
        ProcessingSpeedColumn(),
        "[purple]{task.fields[comment]}[/]",
    )


class DownstreamDbCreator(Generic[DatasetSplitSourceType, InstanceModelType]):
    """Create a DatasetDb file for a downstream dataset."""

    db_file_ext: str = "db"

    def __init__(
        self,
        dataset_name: DatasetName,
        source_per_split: dict[DatasetSplit, DatasetSplitSourceType],
        instance_creator: DownstreamInstanceCreator[InstanceModelType],
        progress: Progress,
        output_dir: Path = settings.paths.databases,
    ) -> None:
        self.dataset_name = dataset_name

        self.source_per_split: dict[DatasetSplit, DatasetSplitSourceType] = source_per_split
        self.instance_creator = instance_creator

        self._output_dir = output_dir

        # Store progress and create tasks for each dataset split
        self.progress = progress
        self._task_ids = self._create_progress_tasks(source_per_split.keys())

    @classmethod
    def from_jsonl(
        cls,
        dataset_name: DatasetName,
        source_per_split: dict[DatasetSplit, Path],
        instance_model_type: type[InstanceModelType],
        output_dir: Path = settings.paths.databases,
    ) -> "DownstreamDbCreator[Path, InstanceModelType]":
        """Create the DatasetDb file using JSONL files from the downstream dataset.

        The entire dataet must be contained within a JSONL file.
        """
        progress = create_downstream_rich_progress()

        if not all(path.suffix.lower().endswith("jsonl") for path in source_per_split.values()):
            raise AssertionError("All provided paths must be `JSONL` files.")

        if not all(path.is_file() for path in source_per_split.values()):
            raise AssertionError("All provided file paths must be a single file.")

        instance_creator = DownstreamInstanceCreator(
            instance_model_type=instance_model_type,
            progress=progress,
            task_description=f"Creating {dataset_name.value} instances",
        )

        db_creator = DownstreamDbCreator[Path, InstanceModelType](
            dataset_name=dataset_name,
            source_per_split=source_per_split,
            output_dir=output_dir,
            instance_creator=instance_creator,
            progress=progress,
        )

        return db_creator

    @classmethod
    def from_huggingface(
        cls,
        huggingface_dataset_identifier: str,
        dataset_name: DatasetName,
        instance_model_type: type[InstanceModelType],
        output_dir: Path = settings.paths.databases,
        hf_auth_token: Optional[Union[bool, str]] = None,
    ) -> "DownstreamDbCreator[Union[Dataset, IterableDataset], InstanceModelType]":
        """Instantiate a DownstreamDbCreator for datasets from the Hugging Face Hub.

        Args:
            huggingface_dataset_identifier (str): The dataset identifier from the Hugging Face Hub
            dataset_name (DatasetName): Enum for the dataset Name
            instance_model_type (type[InstanceModelType]): The non-instantated instance for your
                dataset
            output_dir (Path): The directory where the DB file will be created. Defaults
                to settings.paths.databases.
            hf_auth_token (Optional[Union[bool, str]]): If required, the Hugging Face
                authentication token. Defaults to None.

        Returns:
            Returns a DB creator that handles Hugging Face datasets.
        """
        dataset = load_dataset(huggingface_dataset_identifier, use_auth_token=hf_auth_token)

        generator_per_splits: dict[DatasetSplit, Union[Dataset, IterableDataset]] = (
            # this dataset has predefined splits
            {DatasetSplit[split_id]: split_dataset for split_id, split_dataset in dataset.items()}
            if isinstance(dataset, (DatasetDict, IterableDatasetDict))
            # this dataset does not have any predefined splits so we use it as is
            else {DatasetSplit.train: dataset}
        )

        progress = create_downstream_rich_progress()

        instance_creator = DownstreamInstanceCreator(
            instance_model_type=instance_model_type,
            progress=progress,
            task_description=f"Creating {dataset_name.value} instances",
            data_storage=TorchStorage(),
        )

        db_creator = DownstreamDbCreator[Union[Dataset, IterableDataset], InstanceModelType](
            dataset_name=dataset_name,
            source_per_split=generator_per_splits,
            output_dir=output_dir,
            instance_creator=instance_creator,
            progress=progress,
        )
        return db_creator

    @classmethod
    def from_one_instance_per_dict(
        cls,
        dataset_name: DatasetName,
        source_per_split: dict[DatasetSplit, list[dict[Any, Any]]],
        instance_model_type: type[InstanceModelType],
        output_dir: Path = settings.paths.databases,
    ) -> "DownstreamDbCreator[list[dict[Any, Any]], InstanceModelType]":
        """Instantiate the Db creator when the input data is loaded and processed in a dictionary.

        Each dictionary element is a list of all instances for a dataset split.
        """
        progress = create_downstream_rich_progress()

        instance_creator = DownstreamInstanceCreator(
            instance_model_type=instance_model_type,
            progress=progress,
            task_description=f"Creating {dataset_name.value} instances",
        )

        db_creator = DownstreamDbCreator[list[dict[Any, Any]], InstanceModelType](
            dataset_name=dataset_name,
            source_per_split=source_per_split,
            output_dir=output_dir,
            instance_creator=instance_creator,
            progress=progress,
        )
        return db_creator

    @classmethod
    def from_one_instance_per_json(
        cls,
        dataset_name: DatasetName,
        source_per_split: dict[DatasetSplit, list[Path]],
        instance_model_type: type[InstanceModelType],
        output_dir: Path = settings.paths.databases,
    ) -> "DownstreamDbCreator[list[Path], InstanceModelType]":
        """Instantiate the Db creator when the input data is separated across JSON files.

        Each JSON file must have one instance.
        """
        progress = create_downstream_rich_progress()

        instance_creator = DownstreamInstanceCreator(
            instance_model_type=instance_model_type,
            progress=progress,
            task_description=f"Creating {dataset_name.value} instances",
        )

        db_creator = DownstreamDbCreator[list[Path], InstanceModelType](
            dataset_name=dataset_name,
            source_per_split=source_per_split,
            output_dir=output_dir,
            instance_creator=instance_creator,
            progress=progress,
        )
        return db_creator

    def run(self, num_workers: Optional[int] = None) -> None:
        """Use multiprocessing to create and process all input data across all splits."""
        process_pool = Pool(num_workers)

        with self.progress, process_pool:  # noqa: WPS316
            for split, paths in self.source_per_split.items():
                self.run_for_split(
                    iterable_input_data=self._prepare_input_data_for_instance_creator(paths),
                    dataset_split=split,
                    pool=process_pool,
                    storage_type=self._storage_type,
                )

    def run_for_split(
        self,
        iterable_input_data: Union[Iterable[str], Iterable[Path], Iterable[dict[Any, Any]]],
        dataset_split: DatasetSplit,
        pool: Optional[Pool] = None,
        storage_type: StorageType = StorageType.json,
    ) -> None:
        """Process and write the input data for a given dataset split."""
        task_id = self._task_ids[dataset_split]

        instance_iterator = self.instance_creator(iterable_input_data, self.progress, pool)

        self.progress.reset(task_id, start=True, visible=True)
        with DatasetDb(
            self._get_db_path(dataset_split), readonly=False, storage_type=storage_type
        ) as db:
            for idx, instance in enumerate(instance_iterator):
                dataset_idx = f"{self.dataset_name.name}_{dataset_split.name}_{idx}"

                db[(idx, dataset_idx)] = instance
                self.progress.advance(task_id)

            self._end_progress(task_id)

    def _create_progress_tasks(
        self, dataset_splits: Iterable[DatasetSplit]
    ) -> dict[DatasetSplit, TaskID]:
        """Create tasks on the progress bar for each dataset split."""
        return {
            dataset_split: self.progress.add_task(
                f"Writing {dataset_split.value} instances for {self.dataset_name.value}",
                total=float("inf"),
                start=False,
                visible=False,
                comment="",
            )
            for dataset_split in dataset_splits
        }

    def _get_db_path(self, dataset_split: DatasetSplit) -> Path:
        """Get the output location of the DatasetDb file for a given split."""
        db_file_name = f"{self.dataset_name.name}_{dataset_split.name}.{self.db_file_ext}"
        return self._output_dir.joinpath(db_file_name)

    def _end_progress(self, task_id: TaskID) -> None:
        """Stop the progress bar and make sure to freeze the finished bar."""
        completed = int(self.progress._tasks[task_id].completed)  # noqa: WPS437
        self.progress.update(
            task_id, visible=True, total=completed, completed=completed, comment="Done!"
        )
        self.progress.stop_task(task_id)

    def _prepare_input_data_for_instance_creator(
        self, data_for_dataset_split: DatasetSplitSourceType
    ) -> Union[Iterable[str], Iterable[Path], Iterable[dict[Any, Any]]]:
        """Convert the path data for a dataset split into a supported form."""
        if isinstance(data_for_dataset_split, Path) and data_for_dataset_split.exists():
            with data_for_dataset_split.open() as split_data_file:
                return split_data_file.readlines()

        if isinstance(data_for_dataset_split, Iterable):
            if all(isinstance(element, Path) for element in data_for_dataset_split):
                return data_for_dataset_split
            elif all(isinstance(element, dict) for element in data_for_dataset_split):
                return data_for_dataset_split

        raise NotImplementedError

    @property
    def _storage_type(self) -> StorageType:
        """Get the current storage type used by the DatasetDb.

        This is extracted from within the instance creator.
        """
        return self.instance_creator.storage.storage_type
