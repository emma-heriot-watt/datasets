from pathlib import Path

import pytest
from pydantic import BaseModel

from emma_datasets.datamodels import ActionTrajectory, Instance
from emma_datasets.datamodels.datasets import AlfredImageMetadata, AlfredMetadata
from emma_datasets.db import DatasetDb
from emma_datasets.io import read_json


def test_can_read_instance_from_db(instances_db: DatasetDb) -> None:
    instance_str = instances_db[0]
    assert instance_str

    instance = Instance.parse_raw(instance_str)
    assert isinstance(instance, Instance)


def test_alfred_instance_trajectory_is_correct_object(alfred_instance: Instance) -> None:
    assert alfred_instance
    if alfred_instance.trajectory is not None:
        assert isinstance(alfred_instance.trajectory, BaseModel)
        assert isinstance(alfred_instance.trajectory, ActionTrajectory)
        high_level_actions = alfred_instance.trajectory.high_level_actions
        if high_level_actions is not None:
            planner_action = high_level_actions[0].planner_action
            if planner_action.action == "PickupObject":
                # if we have a pickup object, we are expecting a non-empty field for object/receptacle
                assert planner_action.object_id is not None
                assert planner_action.coordinate_object_id is not None
                assert planner_action.coordinate_receptable_object_id is not None


def test_alfred_instance_metadata_media_and_actions_match(
    alfred_instance: Instance, alfred_annotations: dict[str, Path]
) -> None:
    """Make sure that we keep the correct frames for Alfred instances."""
    assert alfred_instance
    assert alfred_instance.trajectory is not None
    num_actions = len(alfred_instance.trajectory.low_level_actions)
    # metadata are from only from alfred dataset
    metadata = list(alfred_instance.dataset.values())[0]
    if isinstance(metadata.media, list):
        frames = [frame.path.name for frame in metadata.media]  # type: ignore[union-attr]

        assert len(frames) == num_actions

        annotation_file = alfred_annotations[metadata.id]
        images = AlfredMetadata.parse_obj(read_json(annotation_file)).images
        for high_level_action in alfred_instance.trajectory.high_level_actions:  # type: ignore[union-attr]
            test_alfred_last_frame_per_actions(
                high_idx=high_level_action.high_idx, images=images, frames=frames
            )


@pytest.mark.skip(reason="Not a standalone test")
def test_alfred_last_frame_per_actions(
    high_idx: int, images: list[AlfredImageMetadata], frames: list[str]
) -> None:
    """Make sure that we keep the last frame for each low-level action."""
    subgoal_images = [image for image in images if image.high_idx == high_idx]
    for image_index, subgoal_image in enumerate(subgoal_images[:-1]):
        # Check if it's the last image for the low level action
        if subgoal_image.low_idx == subgoal_images[image_index + 1].low_idx:
            continue

        selected_image = frames.pop(0)
        assert subgoal_image.image_name == selected_image

    # Check for the last action
    selected_image = frames.pop(0)
    assert subgoal_images[-1].image_name == selected_image


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
