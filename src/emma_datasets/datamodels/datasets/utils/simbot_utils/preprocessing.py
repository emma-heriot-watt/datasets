from copy import deepcopy
from typing import Any, Optional

from emma_datasets.datamodels.datasets.utils.simbot_utils.ambiguous_data import (
    AmbiguousGotoProcessor,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.data_augmentations import (
    SyntheticLowLevelActionSampler,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    ClarificationTargetExtractor,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotClarificationTypes,
)


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
    ambiguous: bool = False,
    paraphrasable: bool = False,
    vision_augmentation: bool = False,
    cdf_augmentation: bool = False,
    cdf_highlevel_key: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create an instruction dict."""
    action_start_id = instruction["actions"][0]
    action_end_id = instruction["actions"][-1]
    instruction_actions = deepcopy(actions[action_start_id : action_end_id + 1])

    # add the final label for the last action within an instruction
    instruction_actions[-1]["final"] = True

    if clarification_extractor is not None:
        instruction = prepare_instruction_question_answers(clarification_extractor, instruction)

    instruction_dict = {
        "instruction": instruction,
        "actions": instruction_actions,
        "mission_id": mission_id,
        "annotation_id": annotation_id,
        "instruction_id": instruction_id,
        "synthetic": synthetic,
        "ambiguous": ambiguous,
        "paraphrasable": paraphrasable,
        "vision_augmentation": vision_augmentation,
        "cdf_augmentation": cdf_augmentation,
        "cdf_highlevel_key": cdf_highlevel_key,
    }
    return instruction_dict


def instruction_has_spatial_info(instruction_dict: dict[str, Any]) -> bool:
    """Check if an instruction dict has spatial information.

    This check is done both in the raw instruction text and the question answer. It is used to
    filter out look around actions from human instructions.
    """
    question_answers = instruction_dict.get("question_answers", [])
    qa_concatenations = [f"{qa['question']} {qa['answer']}" for qa in question_answers]

    concat_string = " ".join([instruction_dict["instruction"]] + qa_concatenations)

    has_spatial_info = (
        "left" in concat_string
        or "right" in concat_string
        or "behind" in concat_string
        or "front" in concat_string
    )
    return has_spatial_info


def get_action_types_for_instruction(
    instruction_dict: dict[str, Any], actions: list[dict[str, Any]]
) -> list[str]:
    """Get the action types for an instruction."""
    action_start_id = instruction_dict["actions"][0]
    action_end_id = instruction_dict["actions"][-1]
    return [action["type"] for action in actions[action_start_id : action_end_id + 1]]


def instruction_is_goto_room(
    instruction_dict: dict[str, Any], actions: list[dict[str, Any]]
) -> bool:
    """Determine whether the instruction is a goto room instruction."""
    action_types = get_action_types_for_instruction(instruction_dict, actions)
    instruction_actions = instruction_dict["actions"]
    return all(
        [
            len(action_types) == 1,
            action_types[0].lower() == "goto"
            and "officeRoom" in actions[instruction_actions[0]]["goto"]["object"].keys(),
        ]
    )


class TrajectoryInstructionProcessor:
    """Preprocess the instruction instances for the human annotations."""

    def __init__(self, skip_goto_rooms: bool = True, cdf_augmentation: bool = False) -> None:
        self._clarification_target_extractor = ClarificationTargetExtractor()
        self.skip_goto_rooms = skip_goto_rooms
        self.cdf_augmentation = cdf_augmentation

    def run(  # noqa: WPS231
        self,
        human_annotations: list[dict[str, Any]],
        mission_id: str,
        actions: list[dict[str, Any]],
        instruction_idx: int,
        cdf_highlevel_key: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Run the preprocesing."""
        instruction_data = []
        for human_idx, human_annotation in enumerate(human_annotations):

            for instruction in human_annotation["instructions"]:
                if self.skip_goto_rooms and instruction_is_goto_room(instruction, actions):
                    continue
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
                    clarification_extractor=self._clarification_target_extractor,
                    synthetic=False,
                    cdf_augmentation=self.cdf_augmentation,
                    cdf_highlevel_key=cdf_highlevel_key,
                )
                instruction_data.append(instruction_dict)
                instruction_idx += 1
        return instruction_data


class SyntheticIntructionsPreprocessor:
    """Preprocess the instruction instances for the human annotations."""

    def __init__(
        self,
        skip_goto_rooms: bool = True,
        use_synthetic_action_sampler: bool = False,
        num_additional_synthetic_instructions: int = -1,
    ) -> None:
        self.skip_goto_rooms = skip_goto_rooms
        self.use_synthetic_action_sampler = use_synthetic_action_sampler
        self.num_additionalinstructions = num_additional_synthetic_instructions
        self._synthetic_action_sampler = SyntheticLowLevelActionSampler()
        self._ambiguous_goto_processor = AmbiguousGotoProcessor()
        self.total_sampled_actions = 0

    def run(  # noqa: WPS231
        self,
        synthetic_annotations: list[dict[str, Any]],
        mission_id: str,
        actions: list[dict[str, Any]],
        instruction_idx: int,
    ) -> list[dict[str, Any]]:
        """Run the preprocesing."""
        instruction_data = []

        for annot_idx, synthetic_annotation in enumerate(synthetic_annotations):
            for instruction in synthetic_annotation["instructions"]:
                if self.skip_goto_rooms and instruction_is_goto_room(instruction, actions):
                    continue

                instruction_dict = create_instruction_dict(
                    instruction=instruction,
                    actions=actions,
                    mission_id=mission_id,
                    annotation_id=f"synthetic_{annot_idx}",
                    instruction_id=str(instruction_idx),
                    synthetic=True,
                )

                instruction_data.extend(
                    self._ambiguous_goto_processor(
                        instruction_dict=instruction_dict,
                        mission_id=mission_id,
                        action=actions[instruction["actions"][0]],
                    )
                )
                instruction_idx += 1

                add_synthetic_instructions = (
                    self.num_additionalinstructions == -1
                    or self.total_sampled_actions < self.num_additionalinstructions
                )
                if self.use_synthetic_action_sampler and add_synthetic_instructions:
                    instruction_dict = self._synthetic_action_sampler(
                        mission_id=mission_id,
                        annotation_id=f"synthetic_{annot_idx}",
                        instruction_idx=instruction_idx,
                        original_action=actions[instruction["actions"][0]],
                    )

                    instruction_data.append(instruction_dict)
                    instruction_idx += 1

                    self.total_sampled_actions += 1
        return instruction_data
