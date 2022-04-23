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
        high_level_actions = alfred_instance.trajectory.high_level_actions
        if high_level_actions is not None:
            planner_action = high_level_actions[0].planner_action
            if planner_action.action == "PickupObject":
                # if we have a pickup object, we are expecting a non-empty field for object/receptacle
                assert planner_action.object_id is not None
                assert planner_action.coordinate_object_id is not None
                assert planner_action.coordinate_receptable_object_id is not None


def test_can_access_built_pretraining_instances_without_error(
    subset_instances_db: DatasetDb,
) -> None:
    for _, _, instance_str in subset_instances_db:
        new_instance = Instance.parse_raw(instance_str)
        assert isinstance(new_instance, Instance)


def test_access_to_instance_attributes_without_error(subset_instances_db: DatasetDb) -> None:
    for _, _, instance_str in subset_instances_db:
        new_instance = Instance.parse_raw(instance_str)
        assert new_instance.modality
        assert new_instance.features_path
        assert new_instance.source_paths
