import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.constants import DatasetSplit
from emma_datasets.datamodels.datasets import SimBotInstructionInstance
from emma_datasets.datamodels.datasets.utils.simbot_utils.ambiguous_data import (
    AmbiguousGotoProcessor,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.data_augmentations import (
    SyntheticGotoObjectGenerator,
    SyntheticLowLevelActionSampler,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    ClarificationTargetExtractor,
    create_instruction_dict,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    InstructionParaphraser,
)
from emma_datasets.db import DatasetDb


settings = Settings()


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


def load_simbot_instruction_data(  # noqa: WPS231
    filepath: Path,
    sticky_notes_images_json_path: Path,
    augmentation_images_json_path: Path,
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

    ambiguous_goto_processor = AmbiguousGotoProcessor()
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
                instruction_dict = ambiguous_goto_processor(
                    instruction_dict=instruction_dict,
                    mission_id=mission_id,
                    action=actions[instruction["actions"][0]],
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
    instruction_data.extend(
        load_simbot_sticky_note_instruction_data(
            sticky_notes_images_json_path=sticky_notes_images_json_path,
            num_sticky_notes_instructions=num_sticky_notes_instructions,
            synthetic_action_sampler=synthetic_action_sampler,
        )
    )
    instruction_data.extend(
        load_simbot_augmentation_instruction_data(
            augmentation_images_json_path=augmentation_images_json_path
        )
    )

    return instruction_data


def load_simbot_sticky_note_instruction_data(
    sticky_notes_images_json_path: Path,
    num_sticky_notes_instructions: int,
    synthetic_action_sampler: SyntheticLowLevelActionSampler,
) -> list[dict[Any, Any]]:
    """Load sticky note data."""
    with open(sticky_notes_images_json_path) as fp:
        data = json.load(fp)

    sticky_notes_images = data.keys()
    total_sticky_notes_instructions = 0
    instruction_data = []
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


def load_simbot_augmentation_instruction_data(
    augmentation_images_json_path: Path,
) -> list[dict[Any, Any]]:
    """Load the augmentation data."""
    with open(augmentation_images_json_path) as fp:
        data = json.load(fp)
    paraphraser = InstructionParaphraser()
    instruction_data = []
    for _, mission_metadata in data.items():
        instruction_instance = SimBotInstructionInstance(**mission_metadata)
        mission_metadata["instruction"]["instruction"] = paraphraser.from_instruction_instance(
            instruction_instance
        )
        instruction_dict = create_instruction_dict(**mission_metadata)
        instruction_data.append(instruction_dict)

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
                base_dir.joinpath("train_augmentation_instructions.json"),
                num_additional_synthetic_instructions=train_num_additional_synthetic_instructions,
                num_sticky_notes_instructions=train_num_sticky_notes_instructions,
                add_synthetic_goto_instructions=add_synthetic_goto_instructions,
            ),
            DatasetSplit.valid: load_simbot_instruction_data(
                base_dir.joinpath("valid.json"),
                base_dir.joinpath("valid_sticky_notes.json"),
                base_dir.joinpath("valid_augmentation_instructions.json"),
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
        if instruction_instance.ambiguous:
            continue
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
