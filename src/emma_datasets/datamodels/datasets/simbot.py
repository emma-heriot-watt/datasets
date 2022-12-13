import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.constants import DatasetSplit
from emma_datasets.datamodels.datasets import SimBotInstructionInstance
from emma_datasets.datamodels.datasets.alfred import AlfredMetadata
from emma_datasets.datamodels.datasets.utils.simbot_utils.ambiguous_data import (
    AmbiguousGotoProcessor,
    ClarificationFilter,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.data_augmentations import (
    SyntheticGotoObjectGenerator,
    SyntheticLowLevelActionSampler,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    ClarificationTargetExtractor,
    HoldingObject,
    create_instruction_dict,
    get_action_types_for_instruction,
    instruction_has_spatial_info,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    InstructionParaphraser,
)
from emma_datasets.db import DatasetDb
from emma_datasets.io.paths import get_all_file_paths


settings = Settings()
random.seed(42)  # noqa: WPS432


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
    holding_object_processor = HoldingObject()
    total_sampled_synthetic_actions = 0
    instruction_data = []

    for mission_id, mission_annotations in data.items():
        actions = holding_object_processor(mission_annotations["actions"])

        instruction_idx = 0
        for human_idx, human_annotation in enumerate(mission_annotations["human_annotations"]):
            for instruction in human_annotation["instructions"]:
                action_types = get_action_types_for_instruction(instruction, actions)

                # Ignore look around actions that have spatial information
                if instruction_has_spatial_info(instruction) and "Look" in action_types:
                    continue

                # Ignore look around actions if they are the first action in an instruction
                elif action_types[0] == "Look":
                    instruction["actions"] = instruction["actions"][1:]

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
                instruction_dict = synthetic_goto_generator(  # type: ignore[assignment]
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
        mission_metadata["vision_augmentation"] = True
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
                "vision_augmentation": instruction_instance.vision_augmentation,
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


def filter_clarifications(db_path: Path) -> list[dict[Any, Any]]:  # noqa: WPS231
    """Filter simbot clarifications."""
    filtered_instances = []
    db = DatasetDb(db_path)
    qa_filter = ClarificationFilter()
    for _, _, sample in db:
        instruction_instance = SimBotInstructionInstance.parse_raw(sample)
        if instruction_instance.synthetic:
            filtered_instances.append(instruction_instance.dict())
            continue

        if qa_filter.skip_instruction(instruction_instance.instruction.instruction):
            continue
        # Filter the clarifications
        new_question_answers = qa_filter(instruction_instance)
        if new_question_answers is None:
            new_instruction = instruction_instance.instruction.copy(
                update={"question_answers": new_question_answers}
            )
            new_instruction_instance = instruction_instance.copy(
                update={"instruction": new_instruction}
            )
            filtered_instances.append(new_instruction_instance.dict())
        else:
            for qa_pair in new_question_answers:
                new_instruction = instruction_instance.instruction.copy(
                    update={"question_answers": [qa_pair]}
                )
                new_instruction_instance = instruction_instance.copy(
                    update={"instruction": new_instruction}
                )
                filtered_instances.append(new_instruction_instance.dict())
    return filtered_instances


def load_simbot_clarification_annotations(
    base_dir: Path,
    db_file_name: str,
) -> dict[DatasetSplit, Any]:
    """Loads all the SimBot clarifications."""
    train_db = base_dir.joinpath(f"{db_file_name}_{DatasetSplit.train.name}.db")
    valid_db = base_dir.joinpath(f"{db_file_name}_{DatasetSplit.valid.name}.db")
    source_per_split = {
        DatasetSplit.train: filter_clarifications(train_db),
        DatasetSplit.valid: filter_clarifications(valid_db),
    }
    return source_per_split


def generate_planner_data_from_mission_file(filepath: Path) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations for creating SimBot planner data."""
    with open(filepath) as fp:
        data = json.load(fp)

    restructured_data = []

    for mission_id, mission_annotations in data.items():
        human_annotations = mission_annotations["human_annotations"]
        task_description = mission_annotations["CDF"]["task_description"]

        for h_annotation in human_annotations:

            restructured_data.append(
                {
                    "mission_id": mission_id,
                    "task_description": task_description,
                    "instructions": [
                        instruction["instruction"] for instruction in h_annotation["instructions"]
                    ],
                }
            )

    return restructured_data


def generate_planner_data_from_alfred(alfred_path: Path) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations for creating Simbot planner data from ALFRED."""
    all_file_paths = get_all_file_paths(alfred_path.iterdir())
    restructured_data = []

    for file_path in all_file_paths:
        with open(file_path) as in_file:
            metadata_str = json.load(in_file)
            metadata = AlfredMetadata.parse_obj(metadata_str)

            for ann in metadata.turk_annotations["anns"]:
                restructured_data.append(
                    {
                        "mission_id": metadata.task_id,
                        "task_description": ann.task_desc,
                        "instructions": ann.high_descs,
                    }
                )

    return restructured_data


def load_simbot_planner_annotations(
    simbot_base_dir: Path, alfred_data_dir: Path
) -> dict[DatasetSplit, Any]:
    """Generates raw data for the high-level planner to be used for SimBot."""
    train_data = generate_planner_data_from_mission_file(
        simbot_base_dir.joinpath("train.json")
    ) + generate_planner_data_from_alfred(alfred_data_dir.joinpath("train"))

    valid_data = (
        generate_planner_data_from_mission_file(simbot_base_dir.joinpath("valid.json"))
        + generate_planner_data_from_alfred(alfred_data_dir.joinpath("valid_seen"))
        + generate_planner_data_from_alfred(alfred_data_dir.joinpath("valid_unseen"))
    )

    source_per_split = {DatasetSplit.train: train_data, DatasetSplit.valid: valid_data}

    return source_per_split
