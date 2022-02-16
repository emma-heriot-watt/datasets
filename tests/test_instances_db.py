from pathlib import Path
from typing import Iterator

from pytest import fixture, mark

from emma_datasets.common import progress
from emma_datasets.datamodels import Instance
from emma_datasets.datamodels.constants import DatasetName
from emma_datasets.datamodels.dataset_metadata import DatasetMetadata
from emma_datasets.datamodels.datasets.alfred import AlfredHighAction, AlfredLowAction
from emma_datasets.datamodels.trajectory import GenericActionTrajectory
from emma_datasets.db import DatasetDb
from emma_datasets.pipeline import InstanceCreator


def test_read_instance(instances_db: DatasetDb) -> None:

    instance_str = instances_db[0]

    assert instance_str

    instance = Instance.parse_raw(instance_str)

    assert isinstance(instance, Instance)


@fixture
def alfred_instance(instances_db: DatasetDb) -> Instance:
    for (_, _, instance_str) in instances_db:
        instance = Instance.parse_raw(instance_str)

        if DatasetName.alfred in instance.dataset:
            return instance
    return None


def test_read_alfred_instance(alfred_instance: Instance) -> None:
    assert alfred_instance

    if alfred_instance.trajectory is not None:

        assert isinstance(
            alfred_instance.trajectory, GenericActionTrajectory[AlfredLowAction, AlfredHighAction]
        )


@mark.slow
def test_build_instances(
    tmp_path: Path, all_grouped_metadata: Iterator[list[DatasetMetadata]], progress: progress
) -> None:
    creator = InstanceCreator(progress)

    instances = creator(all_grouped_metadata, progress)

    with DatasetDb(tmp_path.joinpath("instances.db"), readonly=False) as write_db:
        for data_idx, instance in enumerate(instances):
            if data_idx == 10:
                break

            write_db[(data_idx, f"pretrain_{data_idx}")] = instance

    # now that we have written some entries let's try and access them
    with DatasetDb(tmp_path.joinpath("instances.db"), readonly=True) as read_db:
        assert len(read_db) == 10

        new_instance = Instance.parse_raw(read_db[0])

        assert isinstance(new_instance, Instance)
