from pathlib import Path
from typing import Iterable, Optional

from rich.progress import Progress

from emma_datasets.common import (
    Settings,
    get_logger,
    get_progress,
    use_rich_for_logging,
    use_rich_for_tracebacks,
)
from emma_datasets.datamodels import DatasetSplit, TeachEdhInstance
from emma_datasets.db import DatasetDb


settings = Settings()
settings.paths.create_dirs()

use_rich_for_logging()
use_rich_for_tracebacks()

logger = get_logger(__name__)


class TeachEdhInstanceDbCreator:
    """Store TEACh EDH Instances within a DatasetDb file."""

    def __init__(
        self,
        edh_instance_file_paths: Iterable[Path],
        dataset_split: DatasetSplit,
        progress: Progress,
        db_file_name_prefix: str = "teach",
        db_file_ext: str = "db",
        output_dir: Path = settings.paths.databases,
    ) -> None:
        self.edh_instance_file_paths = edh_instance_file_paths
        self.dataset_split = dataset_split
        self.progress = progress

        self._output_dir = output_dir
        self._db_file_name_prefix = db_file_name_prefix
        self._db_file_stem = f"{self._db_file_name_prefix}_{self.dataset_split.value}"
        self._db_file_name = f"{self._db_file_stem}.{db_file_ext}"

        self.task_id = progress.add_task(
            f"Creating DB for {dataset_split.value}", total=float("inf")
        )

    @property
    def db_path(self) -> Path:
        """Get the output location of the DatasetDb file."""
        return self._output_dir.joinpath(self._db_file_name)

    def run(self) -> None:
        """Create the DatasetDb and add all the instances to it."""
        db = DatasetDb(self.db_path, readonly=False)

        for idx, instance_path in enumerate(self.edh_instance_file_paths):
            db[(idx, f"{self._db_file_stem}_{idx}")] = TeachEdhInstance.parse_file(instance_path)

            self.progress.advance(self.task_id)


def create_teach_edh_instances(
    teach_edh_instances_splits_path: Path = settings.paths.teach_edh_instances,
    output_dir: Path = settings.paths.databases,
    progress: Optional[Progress] = None,
) -> None:
    """Create DBs for each split of TEACh EDH instances."""
    progress = progress if progress else get_progress()

    edh_instance_dir_paths: dict[DatasetSplit, Path] = {
        DatasetSplit.train: teach_edh_instances_splits_path.joinpath("train"),
        DatasetSplit.valid_seen: teach_edh_instances_splits_path.joinpath("valid_seen"),
        DatasetSplit.valid_unseen: teach_edh_instances_splits_path.joinpath("valid_unseen"),
    }

    all_creators: list[TeachEdhInstanceDbCreator] = []

    with progress:
        for dataset_split, edh_instance_dir_path in edh_instance_dir_paths.items():
            db_creator = TeachEdhInstanceDbCreator(
                edh_instance_file_paths=edh_instance_dir_path.iterdir(),
                dataset_split=dataset_split,
                progress=progress,
                output_dir=output_dir,
            )

            all_creators.append(db_creator)

        for creator in all_creators:
            creator.run()


if __name__ == "__main__":
    create_teach_edh_instances()
