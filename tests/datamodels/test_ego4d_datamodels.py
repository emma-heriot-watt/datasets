from pathlib import Path

from pydantic import parse_obj_as

from emma_datasets.datamodels.datasets.ego4d import (
    Ego4DMomentsInstance,
    Ego4DNLQInstance,
    Ego4DVQInstance,
    load_ego4d_annotations,
)


def test_can_load_ego4d_nlq_data(ego4d_nlq_instances_path: Path) -> None:
    assert ego4d_nlq_instances_path.exists()

    instances = []

    instances = parse_obj_as(
        list[Ego4DNLQInstance], load_ego4d_annotations(ego4d_nlq_instances_path)
    )

    assert instances, "The file doesn't contain any instances."

    parsed_instance = instances[0]

    assert parsed_instance
    assert parsed_instance.video_uid


def test_can_load_ego4d_vq_data(ego4d_vq_instances_path: Path) -> None:
    assert ego4d_vq_instances_path.exists()

    instances = []

    instances = parse_obj_as(
        list[Ego4DVQInstance], load_ego4d_annotations(ego4d_vq_instances_path)
    )

    assert instances, "The file doesn't contain any instances."

    parsed_instance = instances[0]

    assert parsed_instance
    assert parsed_instance.video_uid


def test_can_load_ego4d_moments_data(ego4d_moments_instances_path: Path) -> None:
    assert ego4d_moments_instances_path.exists()

    instances = []

    instances = parse_obj_as(
        list[Ego4DMomentsInstance], load_ego4d_annotations(ego4d_moments_instances_path)
    )

    assert instances, "The file doesn't contain any instances."

    parsed_instance = instances[0]

    assert parsed_instance
    assert parsed_instance.video_uid
