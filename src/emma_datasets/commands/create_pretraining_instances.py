from multiprocessing.pool import Pool
from typing import Optional

from rich.progress import Progress

from emma_datasets.common import (
    Settings,
    get_progress,
    use_rich_for_logging,
    use_rich_for_tracebacks,
)
from emma_datasets.db import DatasetDb
from emma_datasets.pipeline import InstanceCreator, MetadataParser


BATCH_SIZE = 4096
use_rich_for_logging()
use_rich_for_tracebacks()
settings = Settings()
settings.paths.create_dirs()


instances_db_path = settings.paths.databases.joinpath("instances.db")


def create_pretraining_instances(
    num_workers: Optional[int] = None, progress: Optional[Progress] = None
) -> None:
    """Create all the pretraining instances."""
    progress = progress if progress else get_progress()

    with progress:
        metadata_parser = MetadataParser(progress)
        instance_creator = InstanceCreator(progress)

        metadata_groups = metadata_parser.get_all_metadata_groups()

        with DatasetDb(instances_db_path, readonly=False, batch_size=BATCH_SIZE) as db:
            with Pool(num_workers) as pool:
                instances_iterator = instance_creator(metadata_groups, progress, pool)

                for i, instance in enumerate(instances_iterator):
                    db[(i, f"pretrain_{i}")] = instance.json()


if __name__ == "__main__":
    create_pretraining_instances()
