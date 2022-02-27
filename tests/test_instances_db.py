from pydantic import BaseModel

from emma_datasets.datamodels import GenericActionTrajectory, Instance
from emma_datasets.db import DatasetDb


def test_can_read_instance_from_db(instances_db: DatasetDb) -> None:
    instance_str = instances_db[0]
    assert instance_str

    instance = Instance.parse_raw(instance_str)
    assert isinstance(instance, Instance)


def test_alfred_instance_trajectory_is_correct_object(alfred_instance: Instance) -> None:
    assert alfred_instance

    if alfred_instance.trajectory is not None:
        assert isinstance(alfred_instance.trajectory, BaseModel)
        assert isinstance(alfred_instance.trajectory, GenericActionTrajectory)


def test_can_access_built_pretraining_instances_without_error(
    subset_instances_db: DatasetDb,
) -> None:
    for _, _, instance_str in subset_instances_db:
        new_instance = Instance.parse_raw(instance_str)
        assert isinstance(new_instance, Instance)
