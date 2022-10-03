import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

import spacy
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


def load_simbot_instruction_data(filepath: Path) -> list[dict[Any, Any]]:
    """Loads and reformats the SimBot annotations for creating Simbot instructions."""
    with open(filepath) as fp:
        data = json.load(fp)

    clarification_target_extractor = ClarificationTargetExtractor()
    instruction_data = []
    for mission_id, mission_annotations in data.items():
        actions = mission_annotations["actions"]
        human_annotations = mission_annotations["human_annotations"]
        instruction_idx = 0
        for human_idx, human_annotation in enumerate(human_annotations):
            for instruction in human_annotation["instructions"]:
                action_start_id = instruction["actions"][0]
                action_end_id = instruction["actions"][-1]
                instruction_actions = actions[action_start_id : action_end_id + 1]
                instruction = prepare_instruction_question_answers(
                    clarification_target_extractor, instruction
                )
                instruction_dict = {
                    "mission_id": mission_id,
                    "annotation_id": f"human_{human_idx}",
                    "instruction_id": str(instruction_idx),
                    "instruction": instruction,
                    "actions": instruction_actions,
                }
                instruction_data.append(instruction_dict)
                instruction_idx += 1

        synthetic_annotations = mission_annotations["synthetic_annotations"]
        for annot_id, synthetic_annotation in enumerate(synthetic_annotations):
            for instruction in synthetic_annotation["instructions"]:  # noqa: WPS440
                action_start_id = instruction["actions"][0]
                action_end_id = instruction["actions"][-1]
                instruction_actions = actions[action_start_id : action_end_id + 1]
                instruction_dict = {
                    "mission_id": mission_id,
                    "annotation_id": f"synthetic_{annot_id}",
                    "instruction_id": str(instruction_idx),
                    "instruction": instruction,
                    "actions": instruction_actions,
                }
                instruction_data.append(instruction_dict)
                instruction_idx += 1
    return instruction_data


def load_simbot_annotations(
    base_dir: Path,
    annotation_type: Literal["missions", "instructions"] = "missions",
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
