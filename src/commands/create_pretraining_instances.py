from multiprocessing.pool import Pool
from typing import Optional

from rich.progress import Progress

from src.api.storage import DatasetDB
from src.common import Settings, get_progress
from src.pipeline import InstanceCreator, MetadataParser


settings = Settings()


instances_db_path = settings.paths.databases.joinpath("instances.db")


def create_pretraining_instances(
    num_workers: int = 4, progress: Optional[Progress] = None
) -> None:
    """Create all the pretraining instances."""
    progress = progress if progress else get_progress()

    with progress:
        metadata_parser = MetadataParser(progress)
        instance_creator = InstanceCreator(progress)

        metadata_groups = metadata_parser.get_all_metadata_groups()

        with DatasetDB(instances_db_path, readonly=False) as db:
            progress.update(instance_creator.task_id, filepath=instances_db_path)

            with Pool(num_workers) as pool:
                instances_iterator = instance_creator(metadata_groups, progress, pool)

                for i, instance in enumerate(instances_iterator):
                    db[(i, f"pretrain_{i}")] = instance.json()


if __name__ == "__main__":
    create_pretraining_instances()
