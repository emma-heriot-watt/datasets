import json
import logging
import random
from pathlib import Path
from typing import Any, Literal, Optional

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.constants import DatasetSplit
from emma_datasets.datamodels.datasets import SimBotInstructionInstance
from emma_datasets.datamodels.datasets.alfred import AlfredMetadata
from emma_datasets.datamodels.datasets.utils.simbot_utils.ambiguous_data import (
    ClarificationFilter,
    VisionAugmentationFilter,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.data_augmentations import (
    SyntheticLowLevelActionSampler,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    InventoryObjectfromTrajectory,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    InstructionParaphraser,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.preprocessing import (
    SyntheticIntructionsPreprocessor,
    TrajectoryInstructionProcessor,
    create_instruction_dict,
)
from emma_datasets.db import DatasetDb
from emma_datasets.io.paths import get_all_file_paths


settings = Settings()
random.seed(42)  # noqa: WPS432
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def load_simbot_trajectory_instruction_data(
    trajectory_json_path: Path,
    skip_goto_rooms: bool = True,
    use_synthetic_action_sampler: bool = False,
    num_additional_synthetic_instructions: int = -1,
) -> list[dict[Any, Any]]:
    """Loads the SimBot annotations for creating SimBot trajectories."""
    trajectory_instruction_data = []
    with open(trajectory_json_path) as fp:
        data = json.load(fp)

    human_instruction_processor = TrajectoryInstructionProcessor(
        skip_goto_rooms=skip_goto_rooms,
        cdf_augmentation=False,
    )

    synthetic_instruction_processor = SyntheticIntructionsPreprocessor(
        skip_goto_rooms=skip_goto_rooms,
        use_synthetic_action_sampler=use_synthetic_action_sampler,
        num_additional_synthetic_instructions=num_additional_synthetic_instructions,
    )

    inventory_object_processor = InventoryObjectfromTrajectory()
    for mission_id, mission_annotations in data.items():
        actions = inventory_object_processor(mission_annotations["actions"])

        instruction_idx = 0
        # Human annotations
        instruction_dicts = human_instruction_processor.run(
            human_annotations=mission_annotations["human_annotations"],
            mission_id=mission_id,
            actions=actions,
            instruction_idx=instruction_idx,
        )
        trajectory_instruction_data.extend(instruction_dicts)
        instruction_idx += len(instruction_dicts)

        # Synthetic annotations
        instruction_dicts = synthetic_instruction_processor.run(
            synthetic_annotations=mission_annotations["synthetic_annotations"],
            mission_id=mission_id,
            actions=actions,
            instruction_idx=instruction_idx,
        )
        trajectory_instruction_data.extend(instruction_dicts)
    return trajectory_instruction_data


def load_synthetic_trajectory_instruction_data(trajectory_json_path: Path) -> list[dict[Any, Any]]:
    """Loads the annotations for creating synthetic (CDF) trajectories."""
    trajectory_instruction_data = []
    with open(trajectory_json_path) as fp:
        data = json.load(fp)

    trajectory_instruction_processor = TrajectoryInstructionProcessor(
        skip_goto_rooms=False,
        cdf_augmentation=True,
    )
    inventory_object_processor = InventoryObjectfromTrajectory()

    for mission_id, mission_annotations in data.items():
        # T.20230412__action--pickup_target-object--Apple_from-receptacle--FridgeUpper_02_from-receptacle-is-container-citxf_add_gotoFalse
        cdf_highlevel_key = mission_id.split("__")[1].split("_add")[0]

        actions = inventory_object_processor(mission_annotations["actions"])

        instruction_idx = 0
        instruction_dicts = trajectory_instruction_processor.run(
            human_annotations=mission_annotations["human_annotations"],
            mission_id=mission_id,
            actions=actions,
            instruction_idx=instruction_idx,
            cdf_highlevel_key=cdf_highlevel_key,
        )
        trajectory_instruction_data.extend(instruction_dicts)
        instruction_idx += len(instruction_dicts)
    return trajectory_instruction_data


def load_simbot_data(
    simbot_trajectory_json_path: Optional[Path] = None,
    synthetic_trajectory_json_path: Optional[Path] = None,
    sticky_notes_images_json_path: Optional[Path] = None,
    augmentation_images_json_path: Optional[Path] = None,
    annotation_images_json_path: Optional[Path] = None,
    num_additional_synthetic_instructions: int = -1,
    num_sticky_notes_instructions: int = -1,
    skip_goto_rooms: bool = True,
    use_synthetic_action_sampler: bool = False,
) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations for creating Simbot instructions."""
    instruction_data = []

    # SimBot human + synthetic trajectory data
    if simbot_trajectory_json_path is not None and simbot_trajectory_json_path.exists():
        logger.info("Loading SimBot trajectory data")
        instruction_data.extend(
            load_simbot_trajectory_instruction_data(
                trajectory_json_path=simbot_trajectory_json_path,
                skip_goto_rooms=skip_goto_rooms,
                use_synthetic_action_sampler=use_synthetic_action_sampler,
                num_additional_synthetic_instructions=num_additional_synthetic_instructions,
            )
        )

    # Synthetically generated trajectory data
    if synthetic_trajectory_json_path is not None and synthetic_trajectory_json_path.exists():
        logger.info("Loading synthetic CDF trajectory data")
        instruction_data.extend(
            load_synthetic_trajectory_instruction_data(
                trajectory_json_path=synthetic_trajectory_json_path,
            )
        )

    # Sticky Note data
    if sticky_notes_images_json_path is not None and sticky_notes_images_json_path.exists():
        logger.info("Loading sticky note data")
        synthetic_action_sampler = SyntheticLowLevelActionSampler()
        instruction_data.extend(
            load_simbot_sticky_note_instruction_data(
                sticky_notes_images_json_path=sticky_notes_images_json_path,
                num_sticky_notes_instructions=num_sticky_notes_instructions,
                synthetic_action_sampler=synthetic_action_sampler,
            )
        )
    # Augmentation data
    if augmentation_images_json_path is not None and augmentation_images_json_path.exists():
        logger.info("Loading vision augmentation data")
        instruction_data.extend(
            load_simbot_augmentation_instruction_data(
                augmentation_images_json_path=augmentation_images_json_path,
                paraphrase_when_creating_instruction=True,
            )
        )

    # Additional manual annotations data
    if annotation_images_json_path is not None and annotation_images_json_path.exists():
        logger.info("Loading manual annotation data")
        instruction_data.extend(
            load_simbot_augmentation_instruction_data(
                augmentation_images_json_path=annotation_images_json_path,
                paraphrase_when_creating_instruction=False,
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
    augmentation_images_json_path: Path, paraphrase_when_creating_instruction: bool = True
) -> list[dict[Any, Any]]:
    """Load the augmentation data."""
    ambiguity_filter = VisionAugmentationFilter()
    with open(augmentation_images_json_path) as fp:
        data = json.load(fp)
    paraphraser = InstructionParaphraser()
    instruction_data = []
    for _, mission_metadata in data.items():
        if paraphrase_when_creating_instruction:
            instruction_instance = SimBotInstructionInstance(**mission_metadata)
            instruction_instance.vision_augmentation = True
            ambiguous = ambiguity_filter(instruction_instance)
            if ambiguous:
                continue
            (
                instruction,
                inventory_object_id,
            ) = paraphraser.from_instruction_instance(instruction_instance)
            mission_metadata["instruction"]["instruction"] = instruction
            if inventory_object_id is not None:
                mission_metadata["actions"][0]["inventory_object_id"] = inventory_object_id
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
) -> dict[DatasetSplit, Any]:
    """Loads all the SimBot mission annotation files."""
    if annotation_type == "missions":
        source_per_split = {
            DatasetSplit.train: load_simbot_mission_data(base_dir.joinpath("train.json")),
            DatasetSplit.valid: load_simbot_mission_data(base_dir.joinpath("valid.json")),
        }
    else:
        source_per_split = {
            DatasetSplit.train: load_simbot_data(
                simbot_trajectory_json_path=base_dir.joinpath("train.json"),
                synthetic_trajectory_json_path=base_dir.joinpath("train_trajectories.json"),
                # sticky_notes_images_json_path=base_dir.joinpath("train_sticky_notes.json"),
                annotation_images_json_path=base_dir.joinpath(
                    "train_annotation_instructions_v4.2.json"
                ),
                augmentation_images_json_path=base_dir.joinpath(
                    "train_augmentation_images_new_classes_v6.2.json"
                ),
                num_additional_synthetic_instructions=train_num_additional_synthetic_instructions,
                num_sticky_notes_instructions=train_num_sticky_notes_instructions,
            ),
            DatasetSplit.valid: load_simbot_data(
                simbot_trajectory_json_path=base_dir.joinpath("valid.json"),
                # synthetic_trajectory_json_path=base_dir.joinpath("valid_trajectories.json"),
                # sticky_notes_images_json_path=base_dir.joinpath("valid_sticky_notes.json"),
                annotation_images_json_path=base_dir.joinpath(
                    "valid_annotation_instructions_v4.2.json"
                ),
                augmentation_images_json_path=base_dir.joinpath(
                    "valid_augmentation_images_new_classes_v6.2.json"
                ),
                num_additional_synthetic_instructions=valid_num_additional_synthetic_instructions,
                num_sticky_notes_instructions=valid_num_sticky_notes_instructions,
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
                "cdf_augmentation": instruction_instance.cdf_augmentation,
                "cdf_highlevel_key": instruction_instance.cdf_highlevel_key,
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
        # Do not use synthetic trajectory data like "pour it in the coffee maker"
        # We should not be using instructions with pronouns for NLU because the model needs to
        # learn to predict the missing inventory
        if instruction_instance.synthetic:
            if instruction_instance.vision_augmentation:
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
