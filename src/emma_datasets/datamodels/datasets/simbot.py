import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, root_validator

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetSplit, MediaType


settings = Settings()


class SimBotQA(BaseModel):
    """Class that contains the SimBot question answer annotations for a given step."""

    question: str
    answer: str
    question_necessary: bool


class SimBotAction(BaseModel):
    """SimBot action API data structure."""

    class Config:
        """Custom configuration to allows additional fields."""

        extra: str = "allow"

    id: int
    type: str
    color_images: list[str] = Field(..., alias="colorImages")

    @root_validator(pre=True)
    @classmethod
    def check_action_data(cls, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Validates the current action data structure.

        It makes sure that it contains a field corresponding to the action type.
        """
        if data_dict["type"].lower() not in data_dict:
            raise ValueError(f"Action data should have a field for `{data_dict['type']}`")

        return data_dict

    @property
    def get_action_data(self) -> dict[str, Any]:
        """Extracts the field corresponding to the current action data."""
        return getattr(self, self.type.lower())


class SimBotInstruction(BaseModel):
    """SimBot instruction language annotations."""

    instruction: str
    actions: list[int]
    question_answers: Optional[list[SimBotQA]]


class SimBotAnnotation(BaseModel):
    """Represents a sequence of pairs (actions, instruction)."""

    instructions: list[SimBotInstruction]


class SimBotInstance(BaseInstance):
    """A SimBot instance for the trajectory dataset."""

    mission_id: str
    human_annotations: list[SimBotAnnotation]
    synethetic_annotations: Optional[list[SimBotAnnotation]]

    @property
    def modality(self) -> MediaType:
        """Returns the modality for the given instance.

        SimBot has multicam views because of the look-around action which returns 4 images.
        """
        return MediaType.multicam

    @property
    def features_path(self) -> Path:
        """Returns the path to the features for the current mission."""
        return settings.paths.simbot_features.joinpath(f"{self.mission_id}.pt")


def load_simbot_data(filepath: Path) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations."""
    with open(filepath) as in_file:
        data = json.load(in_file)

    restructured_data = []

    for mission_id, mission_annotations in data.items():
        data = {
            "mission_id": mission_id,
        }

        data.update(mission_annotations)

        restructured_data.append(data)

    return restructured_data


def load_simbot_annotations(base_dir: Path) -> dict[DatasetSplit, Any]:
    """Loads all the SimBot annotation files."""
    source_per_split = {
        DatasetSplit.train: load_simbot_data(base_dir.joinpath("train.json")),
        DatasetSplit.valid: load_simbot_data(base_dir.joinpath("valid.json")),
    }

    return source_per_split
