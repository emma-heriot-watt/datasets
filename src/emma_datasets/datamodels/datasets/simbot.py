import json
import random
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import spacy
from pydantic import BaseModel, Field, root_validator

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetSplit, MediaType
from emma_datasets.db import DatasetDb


settings = Settings()

SYNTHETIC_JSON = settings.paths.constants.joinpath("simbot_low_level_actions_templates.json")


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


class SyntheticLowLevelActionSampler:
    """Create synthetic examples of low level actions."""

    def __init__(self, input_json: Path = SYNTHETIC_JSON) -> None:
        with open(input_json) as fp:
            self._low_level_action_templates = json.load(fp)
        # TODO: Examine for now is only reserved for the sticky notes
        # but apparently the examine can be used for any other object as well
        # https://alexaprizesim-ldg5293.slack.com/files/U02SFPST8AK/F043B3MAX1S/arena_for_embodied_ai_-_user_manual.pdf
        self._low_level_actions = [
            key for key in self._low_level_action_templates.keys() if key != "Examine"
        ]

    def __call__(
        self,
        original_action: Optional[dict[str, Any]] = None,
        sample_sticky_note: bool = False,
        sticky_note_image: Optional[str] = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Sample a low level action and an instruction template."""
        if sample_sticky_note:
            if sticky_note_image is None:  # or sticky_note_image_layout is None:
                raise AssertionError("Need a path to a sticky note image and the image layout")

            low_level_action = "Examine"
            low_level_action_template = random.choice(
                self._low_level_action_templates[low_level_action]["templates"]
            )
            synthetic_instruction = {
                "instruction": low_level_action_template,
                "actions": [0],
            }

            action_type = self._low_level_action_templates[low_level_action]["type"]
            synthetic_action = {
                "id": 0,
                "type": action_type,
                action_type.lower(): {  # TODO: populate the mask field, lol GL
                    "object": {"colorImageIndex": 0, "id": "Sticky Note", "mask": []}
                },
                "colorImages": [sticky_note_image],
            }
        else:
            if original_action is None:
                raise AssertionError("Need the original actions")
            low_level_action = random.choice(self._low_level_actions)
            low_level_action_template = random.choice(
                self._low_level_action_templates[low_level_action]["templates"]
            )
            original_action_first_idx = original_action["id"]
            synthetic_instruction = {
                "instruction": low_level_action_template,
                "actions": [original_action_first_idx],
            }
            action_type = self._low_level_action_templates[low_level_action]["type"]
            synthetic_action = {
                "id": original_action_first_idx,
                "type": action_type,
                action_type.lower(): {
                    "direction": self._low_level_action_templates[low_level_action]["direction"]
                },
                "colorImages": original_action["colorImages"],
            }
        return (synthetic_instruction, [synthetic_action])


class ClarificationTargetExtractor:
    """Extract the target noun phrase for the clarfication question.

    Spelling correction did not work for some cases, for which we fix it manually.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:

        self.nlp = spacy.load(spacy_model)

        self.nlp.add_pipe("merge_noun_chunks")
        self._prefer_naive = {"look"}  # The verb 'look' is sometimes confused as a noun
        self._skipped_nouns = {"can", "sink", "floppy"}  # Cannot identify certain words as nouns
        # Rule-based approach to get the target word from its position.
        # This fails for compound words, for which we use spacy noun chunks.
        self.target_index = {
            SimBotClarificationTypes.description: 3,
            SimBotClarificationTypes.disambiguation: 1,
            SimBotClarificationTypes.location: 3,
        }

    def __call__(self, question: str, question_type: SimBotClarificationTypes) -> Optional[str]:
        """Preprocess the clarification target."""
        tokens = question.split()
        target_index = min(self.target_index[question_type], len(tokens) - 1)
        naive_target = self.get_naive_target(tokens, target_index=target_index)
        target = self.get_target(question, target_index=target_index)

        if target is None or naive_target in self._prefer_naive:
            target = naive_target

        return target

    def get_naive_target(self, question_tokens: list[str], target_index: int) -> str:
        """Get the target based on the word position."""
        naive_target = question_tokens[target_index]
        return re.sub(r"[^\w\s]", "", naive_target)

    def get_target(self, question: str, target_index: int) -> Optional[str]:  # noqa: WPS231
        """Apply spell correction and find a noun phrase."""
        doc = self.nlp(question.lower())
        target = None
        for index, token in enumerate(doc):
            if index > target_index and token.is_stop:
                continue
            if token.tag_ in {"NNP", "NN"}:
                target = token.text.replace("which ", "")
                target = target.replace("the ", "")
            elif index == target_index and token.text in self._skipped_nouns:
                target = token.text
            if target is not None:
                break
        return target


class SimBotQA(BaseModel):
    """Class that contains the SimBot question answer annotations for a given step."""

    question: str
    answer: str
    question_necessary: bool
    question_type: Optional[SimBotClarificationTypes] = None
    question_target: Optional[str] = None


def get_question_type(question: str) -> SimBotClarificationTypes:
    """Get the type for a given question."""
    question = question.lower()
    question_types = {qtype.value: qtype for qtype in SimBotClarificationTypes}
    if question.startswith("which"):
        if question.split()[1] == "direction":
            qtype = "which direction"
        else:
            qtype = "which+instruction_noun"
    else:
        qtype = " ".join(question.split()[:2])
    return question_types.get(qtype, SimBotClarificationTypes.other)


def get_question_target(
    clarification_target_extractor: ClarificationTargetExtractor,
    question: str,
    question_type: SimBotClarificationTypes,
) -> Optional[str]:
    """Get the type for a given question."""
    if question_type == SimBotClarificationTypes.other:
        return None
    if question_type == SimBotClarificationTypes.direction:
        return None
    return clarification_target_extractor(question, question_type)


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


def prepare_instruction_question_answers(
    clarification_target_extractor: ClarificationTargetExtractor, instruction: dict[str, Any]
) -> dict[str, Any]:
    """Add question types and targets."""
    if "question_answers" not in instruction:
        return instruction
    for question_answer in instruction["question_answers"]:
        question_answer["question_type"] = get_question_type(question=question_answer["question"])
        question_answer["question_target"] = get_question_target(
            clarification_target_extractor,
            question=question_answer["question"],
            question_type=question_answer["question_type"],
        )
    return instruction


def create_instruction_dict(
    instruction: dict[str, Any],
    actions: list[dict[str, Any]],
    mission_id: str,
    annotation_id: str,
    instruction_id: str,
    clarification_extractor: Optional[ClarificationTargetExtractor] = None,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Create an instruction dict."""
    action_start_id = instruction["actions"][0]
    action_end_id = instruction["actions"][-1]
    instruction_actions = actions[action_start_id : action_end_id + 1]
    if clarification_extractor is not None:
        instruction = prepare_instruction_question_answers(clarification_extractor, instruction)

    instruction_dict = {
        "instruction": instruction,
        "actions": instruction_actions,
        "mission_id": mission_id,
        "annotation_id": annotation_id,
        "instruction_id": instruction_id,
        "synthetic": synthetic,
    }
    return instruction_dict


def load_simbot_instruction_data(  # noqa: WPS210, WPS231
    filepath: Path,
    sticky_notes_json_path: Path,
    num_additional_synthetic_instructions: int = -1,
    num_sticky_notes_instructions: int = -1,
) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations for creating Simbot instructions."""
    with open(filepath) as fp:
        data = json.load(fp)

    clarification_target_extractor = ClarificationTargetExtractor()
    synthetic_action_sampler = SyntheticLowLevelActionSampler()
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

                    (synth_instr, synth_action) = synthetic_action_sampler(
                        original_action=actions[instruction["actions"][0]]
                    )
                    instruction_dict = create_instruction_dict(
                        instruction=synth_instr,
                        actions=synth_action,
                        mission_id=mission_id,
                        annotation_id=f"synthetic_{annot_idx}",
                        instruction_id=str(instruction_idx),
                        synthetic=True,
                    )

                    instruction_data.append(instruction_dict)
                    instruction_idx += 1

                    total_sampled_synthetic_actions += 1

    with open(sticky_notes_json_path) as fp:  # noqa: WPS440
        data = json.load(fp)

    sticky_notes_images = data.keys()
    total_sticky_notes_instructions = 0
    for idx, sticky_note_image in enumerate(sticky_notes_images):
        if total_sticky_notes_instructions == num_sticky_notes_instructions:
            break
        (synth_instr, synth_action) = synthetic_action_sampler(
            sample_sticky_note=True,
            sticky_note_image=sticky_note_image,
        )
        instruction_dict = create_instruction_dict(
            instruction=synth_instr,
            actions=synth_action,
            mission_id=Path(sticky_note_image).stem,
            annotation_id=f"synthetic_sticky_note{idx}",
            instruction_id=str(idx),
            synthetic=True,
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
            ),
            DatasetSplit.valid: load_simbot_instruction_data(
                base_dir.joinpath("valid.json"),
                base_dir.joinpath("valid_sticky_notes.json"),
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
