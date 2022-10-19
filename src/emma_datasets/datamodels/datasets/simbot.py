import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, root_validator

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetSplit, MediaType
from emma_datasets.datamodels.datasets.utils.simbot_data_augmentations import (
    SyntheticGotoObjectGenerator,
    SyntheticLowLevelActionSampler,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils import (
    ClarificationTargetExtractor,
    SimBotClarificationTypes,
    create_instruction_dict,
)
from emma_datasets.db import DatasetDb


settings = Settings()


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


def load_simbot_instruction_data(  # noqa: WPS210, WPS231
    filepath: Path,
    sticky_notes_images_json_path: Path,
    num_additional_synthetic_instructions: int = -1,
    num_sticky_notes_instructions: int = -1,
    add_synthetic_goto_instructions: bool = True,
) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations for creating Simbot instructions."""
    with open(filepath) as fp:
        data = json.load(fp)

    clarification_target_extractor = ClarificationTargetExtractor()
    synthetic_action_sampler = SyntheticLowLevelActionSampler()
    if add_synthetic_goto_instructions:
        synthetic_goto_generator = SyntheticGotoObjectGenerator()
    else:
        synthetic_goto_generator = None
    total_sampled_synthetic_actions = 0
    instruction_data = []

    for mission_id, mission_annotations in data.items():
        actions = mission_annotations["actions"]
        instruction_idx = 0
        for human_idx, human_annotation in enumerate(mission_annotations["human_annotations"]):
            for instruction in human_annotation["instructions"]:
                instruction_dict = create_instruction_dict(
                    instruction=instruction,
                    actions=actions,
                    mission_id=mission_id,
                    annotation_id=str(human_idx),
                    instruction_id=str(instruction_idx),
                    clarification_extractor=clarification_target_extractor,
                    synthetic=False,
                )

                instruction_data.append(instruction_dict)
                instruction_idx += 1
                if human_idx > 0 or not synthetic_goto_generator:
                    continue
                instruction_dict = synthetic_goto_generator(
                    mission_id=mission_id,
                    instruction_idx=instruction_idx,
                    instruction_actions=deepcopy(
                        instruction_dict["actions"],
                    ),
                )
                if instruction_dict is not None:
                    instruction_data.append(instruction_dict)
                    instruction_idx += 1

        for annot_idx, synthetic_annotation in enumerate(  # noqa: WPS352
            mission_annotations["synthetic_annotations"]
        ):
            for instruction in synthetic_annotation["instructions"]:  # noqa: WPS440
                instruction_dict = create_instruction_dict(
                    instruction=instruction,
                    actions=actions,
                    mission_id=mission_id,
                    annotation_id=f"synthetic_{annot_idx}",
                    instruction_id=str(instruction_idx),
                    synthetic=True,
                )

                instruction_data.append(instruction_dict)
                instruction_idx += 1

                if (  # noqa: WPS337
                    num_additional_synthetic_instructions == -1
                    or total_sampled_synthetic_actions < num_additional_synthetic_instructions
                ):

                    instruction_dict = synthetic_action_sampler(
                        mission_id=mission_id,
                        annotation_id=f"synthetic_{annot_idx}",
                        instruction_idx=instruction_idx,
                        original_action=actions[instruction["actions"][0]],
                    )

                    instruction_data.append(instruction_dict)
                    instruction_idx += 1

                    total_sampled_synthetic_actions += 1

    with open(sticky_notes_images_json_path) as fp:  # noqa: WPS440
        data = json.load(fp)

    sticky_notes_images = data.keys()
    total_sticky_notes_instructions = 0
    for idx, sticky_note_image in enumerate(sticky_notes_images):
        if total_sticky_notes_instructions == num_sticky_notes_instructions:
            break
        instruction_dict = synthetic_action_sampler(
            mission_id=Path(sticky_note_image).stem,
            annotation_id=f"synthetic_sticky_note{idx}",
            instruction_idx=idx,
            sample_sticky_note=True,
            sticky_note_image=sticky_note_image,
            sticky_note_bbox_coords=data[sticky_note_image]["coords"],
        )
        instruction_data.append(instruction_dict)
        total_sticky_notes_instructions += 1
    return instruction_data


def load_simbot_annotations(
    base_dir: Path,
    annotation_type: Literal["missions", "instructions"] = "missions",
    train_num_additional_synthetic_instructions: int = 20000,
    valid_num_additional_synthetic_instructions: int = -1,
    train_num_sticky_notes_instructions: int = 20000,
    valid_num_sticky_notes_instructions: int = -1,
    add_synthetic_goto_instructions: bool = True,
) -> dict[DatasetSplit, Any]:
    """Loads all the SimBot mission annotation files."""
    if annotation_type == "missions":
        source_per_split = {
            DatasetSplit.train: load_simbot_mission_data(base_dir.joinpath("train.json")),
            DatasetSplit.valid: load_simbot_mission_data(base_dir.joinpath("valid.json")),
        }
    else:
        source_per_split = {
            DatasetSplit.train: load_simbot_instruction_data(
                base_dir.joinpath("train.json"),
                base_dir.joinpath("train_sticky_notes.json"),
                num_additional_synthetic_instructions=train_num_additional_synthetic_instructions,
                num_sticky_notes_instructions=train_num_sticky_notes_instructions,
                add_synthetic_goto_instructions=add_synthetic_goto_instructions,
            ),
            DatasetSplit.valid: load_simbot_instruction_data(
                base_dir.joinpath("valid.json"),
                base_dir.joinpath("valid_sticky_notes.json"),
                num_additional_synthetic_instructions=valid_num_additional_synthetic_instructions,
                num_sticky_notes_instructions=valid_num_sticky_notes_instructions,
                add_synthetic_goto_instructions=add_synthetic_goto_instructions,
            ),
        }

    return source_per_split


def unwrap_instructions(db_path: Path) -> list[dict[Any, Any]]:
    """Unwrap simbot instructions to action-level instances."""
    unwrapped_instances = []
    db = DatasetDb(db_path)
    for _, _, sample in db:
        instruction_instance = SimBotInstructionInstance.parse_raw(sample)
        for action_index, action in enumerate(instruction_instance.actions):
            instruction = instruction_instance.instruction.copy(
                update={"actions": instruction_instance.instruction.actions[: action_index + 1]}
            )

            instruction_dict = {
                "mission_id": instruction_instance.mission_id,
                "annotation_id": f"{instruction_instance.annotation_id}_{action.id}",
                "instruction_id": instruction_instance.instruction_id,
                "instruction": instruction,
                "actions": instruction_instance.actions[: action_index + 1],
                "synthetic": instruction_instance.synthetic,
            }
            unwrapped_instances.append(instruction_dict)
    return unwrapped_instances


def load_simbot_action_annotations(
    base_dir: Path,
    db_file_name: str,
) -> dict[DatasetSplit, Any]:
    """Loads all the SimBot actions."""
    train_db = base_dir.joinpath(f"{db_file_name}_{DatasetSplit.train.name}.db")
    valid_db = base_dir.joinpath(f"{db_file_name}_{DatasetSplit.valid.name}.db")
    source_per_split = {
        DatasetSplit.train: unwrap_instructions(train_db),
        DatasetSplit.valid: unwrap_instructions(valid_db),
    }

    return source_per_split
