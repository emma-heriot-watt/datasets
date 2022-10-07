from pathlib import Path

from pydantic import parse_obj_as

from emma_datasets.datamodels.datasets import SimBotInstructionInstance, SimBotMissionInstance
from emma_datasets.datamodels.datasets.simbot import (
    load_simbot_instruction_data,
    load_simbot_mission_data,
)


def test_can_load_simbot_mission_data(simbot_instances_path: Path) -> None:
    assert simbot_instances_path.exists()

    instances = parse_obj_as(
        list[SimBotMissionInstance],
        load_simbot_mission_data(simbot_instances_path),
    )
    assert instances, "The file doesn't contain any instances."

    for parsed_instance in instances:
        assert parsed_instance
        assert parsed_instance.mission_id


def test_can_load_simbot_instruction_data(
    simbot_instances_path: Path, simbot_sticky_notes_path: Path
) -> None:
    assert simbot_instances_path.exists()

    instances = parse_obj_as(
        list[SimBotInstructionInstance],
        load_simbot_instruction_data(simbot_instances_path, simbot_sticky_notes_path),
    )
    assert instances, "The file doesn't contain any instances."

    for parsed_instance in instances:
        assert parsed_instance
        assert parsed_instance.mission_id
