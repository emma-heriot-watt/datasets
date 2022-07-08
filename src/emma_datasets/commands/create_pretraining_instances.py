from multiprocessing.pool import Pool
from typing import Optional

from rich.progress import Progress

from emma_datasets.common import Settings, get_progress, use_rich_for_logging
from emma_datasets.db import DatasetDb
from emma_datasets.parsers.instance_creators import PretrainInstanceCreator
from emma_datasets.pipeline import MetadataParser


BATCH_SIZE = 4096
use_rich_for_logging()
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
        instance_creator = PretrainInstanceCreator(progress, should_compress=True)

        metadata_groups = metadata_parser.get_all_metadata_groups()

        db = DatasetDb(instances_db_path, readonly=False, batch_size=BATCH_SIZE)
        process_pool = Pool(num_workers)

        with db, process_pool:  # noqa: WPS316
            instances_iterator = instance_creator(metadata_groups, progress, process_pool)

            for i, instance in enumerate(instances_iterator):
                db[(i, f"pretrain_{i}")] = instance


if __name__ == "__main__":
    create_pretraining_instances()
