import re
from copy import deepcopy
from enum import Enum
from typing import Any, Optional

import spacy


def get_object_from_action_object_metadata(
    object_asset: str, object_assets_to_names: dict[str, str]
) -> str:
    """Map the object asset for a given action to its readable name.

    Example:
        (object_asset, object_name) = (Desk_01_1000, Desk)
    """
    # Case1: Object asset in action matches exactly with object assets
    object_name_candidate = object_assets_to_names.get(object_asset, None)
    if object_name_candidate is not None:
        return object_name_candidate

    # Case2: The object asset in action contains a substring that matches with the object assests
    # Example: Desk_01_1000
    # Because the assets can have additional tags we need to remove these tags
    # and check if they asset after removing the tags match an object label
    object_asset_components = object_asset.split("_")

    for idx in range(len(object_asset_components), 0, -1):
        # tries to match the longest sub-string first
        object_name_candidate = "_".join(object_asset_components[:idx])
        object_name_candidate = object_assets_to_names.get(object_name_candidate, None)
        if object_name_candidate is not None:
            return object_name_candidate
    return object_asset


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
    }
    return instruction_dict
