from multiprocessing.pool import Pool
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
from emma_datasets.datamodels import DatasetSplit
from emma_datasets.db import DatasetDb
from emma_datasets.parsers.instance_creators.teach_edh import TeachEdhInstanceCreator


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

        self.instance_creator = TeachEdhInstanceCreator(
            self.progress,
            task_description=f"Create instances for {dataset_split.value}",
            should_compress=True,
        )

        self.task_id = progress.add_task(
            f"Writing {dataset_split.value} instances to DB",
            total=float("inf"),
            start=False,
            visible=False,
        )

    @property
    def db_path(self) -> Path:
        """Get the output location of the DatasetDb file."""
        return self._output_dir.joinpath(self._db_file_name)

    def run(self, pool: Optional[Pool] = None) -> None:
        """Create the DatasetDb and add all the instances to it."""
        db = DatasetDb(self.db_path, readonly=False)

        instance_iterator = self.instance_creator(
            self.edh_instance_file_paths, self.progress, pool
        )

        self.progress.reset(self.task_id, start=True, visible=True)

        with db:
            for idx, instance in enumerate(instance_iterator):
                db[(idx, f"{self._db_file_stem}_{idx}")] = instance
                self.progress.advance(self.task_id)


def create_teach_edh_instances(
    teach_edh_instances_splits_path: Path = settings.paths.teach_edh_instances,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
    progress: Optional[Progress] = None,
) -> None:
    """Create DBs for each split of TEACh EDH instances."""
    progress = progress if progress else get_progress()
    process_pool = Pool(num_workers)

    edh_instance_dir_paths: dict[DatasetSplit, Path] = {
        DatasetSplit.train: teach_edh_instances_splits_path.joinpath("train"),
        DatasetSplit.valid_seen: teach_edh_instances_splits_path.joinpath("valid_seen"),
        DatasetSplit.valid_unseen: teach_edh_instances_splits_path.joinpath("valid_unseen"),
    }

    all_creators: list[TeachEdhInstanceDbCreator] = []

    with progress, process_pool:  # noqa: WPS316
        for dataset_split, edh_instance_dir_path in edh_instance_dir_paths.items():
            db_creator = TeachEdhInstanceDbCreator(
                edh_instance_file_paths=edh_instance_dir_path.iterdir(),
                dataset_split=dataset_split,
                progress=progress,
                output_dir=output_dir,
            )

            all_creators.append(db_creator)

        for creator in all_creators:
            creator.run(process_pool)


if __name__ == "__main__":
    create_teach_edh_instances()
