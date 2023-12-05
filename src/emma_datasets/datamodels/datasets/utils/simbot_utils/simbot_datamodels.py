from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, root_validator

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import MediaType


settings = Settings()

ParaphrasableActions = {
    "goto",
    "toggle",
    "open",
    "close",
    "pickup",
    "place",
    "search",
    "pour",
    "fill",
    "clean",
    "scan",
    "break",
}


class SimBotClarificationTypes(Enum):
    """SimBot question clarification types.

    The 4 defined question types correspond to the synthetic clarification questions in the
    annotations.
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
        allow_population_by_field_name = True

    id: int
    type: str
    color_images: list[str] = Field(..., alias="colorImages")
    inventory_object_id: Optional[str] = None
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

    @property
    def necessary_question_answers(self) -> list[SimBotQA]:
        """Get the necessary question-answers."""
        necessary_question_answers: list[SimBotQA] = []
        if not self.question_answers:
            return necessary_question_answers
        for qa_pair in self.question_answers:
            if qa_pair.question_necessary:
                necessary_question_answers.append(qa_pair)
        return necessary_question_answers


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
    vision_augmentation: bool = False
    cdf_augmentation: bool = False
    cdf_highlevel_key: Optional[str] = None

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
        """Returns the path to the features for the current instruction.

        Instances comming from vision augmentations have only a single action. Because images can
        belong to multiple instances, to avoid duplicates the feature path is directly the path to
        the image.
        """
        # The instance comes from the vision data augmentations
        if self.vision_augmentation:
            template = "{feature_path}.pt"
            color_image = self.actions[0].color_images[0]
            feature_path = Path(color_image).stem
            return [
                settings.paths.simbot_features.joinpath(template.format(feature_path=feature_path))
            ]

        # The instance comes from the cdf augmentations
        elif self.cdf_augmentation:
            template = "{feature_path}.pt"
            color_images = [action.color_images[0] for action in self.actions]
            feature_paths = [Path(color_image).stem for color_image in color_images]
            return [
                settings.paths.simbot_features.joinpath(template.format(feature_path=feature_path))
                for feature_path in feature_paths
            ]

        # The instance comes from the simbot annotations
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
        # All instances comming from CDF augmentations are paraphrasable
        if self.cdf_augmentation:
            return True

        cond1 = len(self.actions) == 1  # number of actions
        cond2 = self.actions[0].type.lower() in ParaphrasableActions  # action type
        cond3 = self.synthetic  # synthetic and not Goto room
        # Synthetic goto room instructions are not paraphrasable
        action_metadata = self.actions[0].get_action_data.get("object", None)
        if cond3 and action_metadata is not None and "id" not in action_metadata:
            cond3 = False
        return cond1 and cond2 and cond3


class SimBotObjectAttributes(BaseModel):
    """Base model for attributes of objects."""

    readable_name: str
    color: Optional[str] = None
    location: Optional[Literal["left", "middle", "right"]] = None
    distance: Optional[float] = None


class AugmentationInstruction(BaseModel):
    """Basemodel for an augmentation instruction."""

    action_type: str
    object_id: Union[str, list[str]]
    bbox: Union[list[int], list[list[int]], None]  # one, multiple, or no bounding boxes at all
    image_name: str
    attributes: Union[SimBotObjectAttributes, list[SimBotObjectAttributes]]
    annotation_id: int
    image_index: int = 0
    room_name: Optional[str] = None
    augmentation_metadata: Optional[dict[str, Any]] = None


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
