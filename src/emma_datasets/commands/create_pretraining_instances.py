from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import Pool
from typing import Optional

from rich.progress import Progress

from emma_datasets.common import (
    Settings,
    get_progress,
    use_rich_for_logging,
    use_rich_for_tracebacks,
)
from emma_datasets.datamodels import Instance
from emma_datasets.db import DatasetDb
from emma_datasets.pipeline import InstanceCreator, MetadataParser


BATCH_SIZE = 4096
use_rich_for_logging()
use_rich_for_tracebacks()
settings = Settings()
settings.paths.create_dirs()


instances_db_path = settings.paths.databases.joinpath("instances.db")


def write_to_db(db: DatasetDb, idx: int, instance: Instance) -> None:
    """Send the instance to be written to the database."""
    db[(idx, f"pretrain_{idx}")] = instance


def create_pretraining_instances(
    num_workers: Optional[int] = None, progress: Optional[Progress] = None
) -> None:
    """Create all the pretraining instances."""
    progress = progress if progress else get_progress()

    with progress:
        metadata_parser = MetadataParser(progress)
        instance_creator = InstanceCreator(progress)

        metadata_groups = metadata_parser.get_all_metadata_groups()

        db = DatasetDb(instances_db_path, readonly=False, batch_size=BATCH_SIZE)
        process_pool = Pool(num_workers)
        thread_pool = ThreadPoolExecutor()

        with db, process_pool, thread_pool:  # noqa: WPS316
            instances_iterator = instance_creator(metadata_groups, progress, process_pool)

            for i, instance in enumerate(instances_iterator):
                thread_pool.submit(write_to_db, db, i, instance)


if __name__ == "__main__":
    create_pretraining_instances()
