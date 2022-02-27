from pathlib import Path
from typing import Iterator, Optional

from pytest_cases import fixture
from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName, Instance
from emma_datasets.db import DatasetDb
from emma_datasets.pipeline import InstanceCreator


def create_subset_instances_db(
    cached_instances_db_path: Path,
    all_grouped_metadata: Iterator[list[DatasetMetadata]],
    progress: Progress,
) -> bool:
    creator = InstanceCreator(progress)
    all_instances = list(creator(all_grouped_metadata, progress))

    with DatasetDb(cached_instances_db_path, readonly=False) as write_db:
        for data_idx, instance in enumerate(all_instances):
            write_db[(data_idx, f"pretrain_{data_idx}")] = instance

        assert len(write_db) == len(all_instances)

    return True


@fixture
def subset_instances_db(
    cached_instances_db_path: Path,
    all_grouped_metadata: Iterator[list[DatasetMetadata]],
    progress: Progress,
) -> DatasetDb:
    if not cached_instances_db_path.exists():
        create_subset_instances_db(cached_instances_db_path, all_grouped_metadata, progress)

    return DatasetDb(cached_instances_db_path, readonly=True)


@fixture
def alfred_instance(subset_instances_db: DatasetDb) -> Optional[Instance]:
    for (_, _, instance_str) in subset_instances_db:
        instance = Instance.parse_raw(instance_str)

        if DatasetName.alfred in instance.dataset:
            return instance

    return None


@fixture
def instances_db(pretrain_instances_db_path: Path) -> DatasetDb:
    return DatasetDb(pretrain_instances_db_path)
