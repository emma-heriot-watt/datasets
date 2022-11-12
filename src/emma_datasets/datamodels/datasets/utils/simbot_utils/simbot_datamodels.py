from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, root_validator

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import MediaType


settings = Settings()

ParaphrasableActions = {"goto", "toggle", "open", "close", "pickup", "place"}


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
    question_type: Optional[SimBotClarificationTypes] = None
    question_target: Optional[str] = None


class SimBotAction(BaseModel):
    """SimBot action API data structure."""

    class Config:
        """Custom configuration to allows additional fields."""

        extra: str = "allow"

    id: int
    type: str
    color_images: list[str] = Field(..., alias="colorImages")
    final: Optional[bool] = False

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
    annotation_id: str
    instruction_id: str
    instruction: SimBotInstruction
    actions: list[SimBotAction]
    synthetic: bool = False
    ambiguous: bool = False
    keep_only_target_frame: bool = False

    class Config:
        """Custom configuration to allows additional fields."""

        extra: str = "allow"

    @property
    def modality(self) -> MediaType:
        """Returns the modality for the given instance.

        SimBot has multicam views because of the look-around action which returns 4 images.
        """
        return MediaType.multicam

    @property
    def features_path(self) -> list[Path]:
        """Returns the path to the features for the current instruction."""
        template = "{mission_id}_action{action_id}.pt"
        return [
            settings.paths.simbot_features.joinpath(
                template.format(mission_id=self.mission_id, action_id=action.id)
            )
            for action in self.actions
        ]

    @property
    def paraphrasable(self) -> bool:
        """Check if the instance allows for paraphrasing."""
        cond1 = len(self.actions) == 1  # number of actions
        cond2 = self.actions[0].type in ParaphrasableActions  # action type
        return cond1 and cond2 and self.synthetic


class SimBotObjectAttributes(BaseModel):
    """Base model for attributes of objects."""

    readable_name: str
    color: Optional[str] = None
    location: Optional[Literal["left", "right"]] = None


class AugmentationInstruction(BaseModel):
    """Basemodel for an augmentation instruction."""

    action_type: str
    object_id: str
    bbox: list[int]
    image_name: str
    attributes: SimBotObjectAttributes


class SimBotPlannerInstance(BaseInstance):
    """Basemodel for the high-level planner dataset."""

    mission_id: str
    task_description: str
    instructions: list[str]

    @property
    def modality(self) -> MediaType:
        """Returns the modality for the given instance.

        SimBot has multicam views because of the look-around action which returns 4 images.
        """
        return MediaType.multicam

    @property
    def features_path(self) -> list[Path]:
        """Returns the path to the features for the current instruction."""
        return []
