from pathlib import Path

from pydantic import parse_obj_as

from emma_datasets.datamodels.datasets import SimBotInstance
from emma_datasets.datamodels.datasets.simbot import load_simbot_data


def test_can_load_simbot_data(simbot_instances_path: Path) -> None:
    assert simbot_instances_path.exists()

    instances = parse_obj_as(list[SimBotInstance], load_simbot_data(simbot_instances_path))
    assert instances, "The file doesn't contain any instances."

    for parsed_instance in instances:
        assert parsed_instance
        assert parsed_instance.mission_id
