import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, root_validator

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetSplit, MediaType


settings = Settings()


class SimBotClarificationTypes(Enum):
    """SimBot question clarification types.

    The 4 defined question types correspond to the synthetic clarification questions in the annotations.
    https://us-east-1.console.aws.amazon.com/codesuite/codecommit/repositories/AlexaSimbotMLToolbox/browse/refs/heads/main/--/AlexaSimbotToolbox/data/trajectory-data?region=us-east-1
    https://app.slack.com/client/T02SWBF7J7M/C03UQQM3HN0
    """

    location = "where is"
    description = "what does"
    disambiguation = "which+instruction_noun"
    direction = "which direction"
    other = "other"


class SimBotQA(BaseModel):
    """Class that contains the SimBot question answer annotations for a given step."""

    question: str
    answer: str
    question_necessary: bool

    @property
    def question_type(self) -> str:
        """Get the type for a given question."""
        question = self.question.lower()
        question_types = {qtype.value: qtype.name for qtype in SimBotClarificationTypes}
        if question.startswith("which"):
            if question.split()[1] == "direction":
                qtype = "which direction"
            else:
                qtype = "which+instruction_noun"
        else:
            qtype = " ".join(question.split()[:2])
        return question_types.get(qtype, SimBotClarificationTypes.other.name)


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


class SimBotMissionInstance(BaseInstance):
    """A SimBot instance for the mission dataset."""

    mission_id: str
    human_annotations: list[SimBotAnnotation]
    synethetic_annotations: Optional[list[SimBotAnnotation]]
    actions: list[SimBotAction]

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


class SimBotInstructionInstance(BaseInstance):
    """A SimBot instance for the mission dataset."""

    mission_id: str
    human_id: str
    instruction_id: str
    instruction: SimBotInstruction
    actions: list[SimBotAction]

    @property
    def modality(self) -> MediaType:
        """Returns the modality for the given instance.

        SimBot has multicam views because of the look-around action which returns 4 images.
        """
        return MediaType.multicam

    @property
    def features_path(self) -> Path:
        """Returns the path to the features for the current instruction."""
        basename = f"{self.mission_id}_instruction{self.instruction_id}.pt"
        return settings.paths.simbot_features.joinpath(basename)


def load_simbot_mission_data(filepath: Path) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations for creating SimBot missions."""
    with open(filepath) as fp:
        data = json.load(fp)

    restructured_data = []

    for mission_id, mission_annotations in data.items():
        data = {
            "mission_id": mission_id,
        }

        data.update(mission_annotations)

        restructured_data.append(data)

    return restructured_data


def load_simbot_instruction_data(filepath: Path) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations for creating Simbot instructions."""
    with open(filepath) as fp:
        data = json.load(fp)

    instruction_data = []
    for mission_id, mission_annotations in data.items():
        actions = mission_annotations["actions"]
        human_annotations = mission_annotations["human_annotations"]
        for human_idx, human_annotation in enumerate(human_annotations):
            for instruction_idx, instruction in enumerate(human_annotation["instructions"]):
                action_start_id = instruction["actions"][0]
                action_end_id = instruction["actions"][-1]
                instruction_actions = actions[action_start_id : action_end_id + 1]
                instruction_dict = {
                    "mission_id": mission_id,
                    "human_id": str(human_idx),
                    "instruction_id": str(instruction_idx),
                    "instruction": instruction,
                    "actions": instruction_actions,
                }
                instruction_data.append(instruction_dict)
    return instruction_data


def load_simbot_annotations(
    base_dir: Path, annotation_type: Literal["missions", "instructions"] = "missions"
) -> dict[DatasetSplit, Any]:
    """Loads all the SimBot mission annotation files."""
    load_fn = (
        load_simbot_mission_data if annotation_type == "missions" else load_simbot_instruction_data
    )
    source_per_split = {
        DatasetSplit.train: load_fn(base_dir.joinpath("train.json")),
        DatasetSplit.valid: load_fn(base_dir.joinpath("valid.json")),
    }

    return source_per_split
