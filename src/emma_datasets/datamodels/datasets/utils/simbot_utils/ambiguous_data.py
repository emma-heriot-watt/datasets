from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import torch

from emma_datasets.common.settings import Settings
from emma_datasets.constants.simbot.simbot import get_arena_definitions, get_class_thresholds
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    ClarificationTargetExtractor,
    get_object_label_from_object_id,
    get_object_readable_name_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.object_features_processing import (
    ObjectClassDecoder,
    compute_bbox_area,
    compute_bbox_center_coords,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotAction,
    SimBotClarificationTypes,
    SimBotInstructionInstance,
    SimBotQA,
)


settings = Settings()


class AmbiguityProcessor:
    """Preprocess ambiguous instructions.

    An instruction is considered ambiguous if more than one objects of the object class have been
    detected in the same frame or across frames.
    """

    def __init__(self) -> None:
        self.object_decoder = ObjectClassDecoder()
        self._class_thresholds = get_class_thresholds()
        self._default_min_area = 250
        self._min_center_coord = 110
        self._max_center_coord = 180
        self._area_percentage = 0.8

    def ambiguous_across_frames(
        self,
        frame_index: int,
        features_path: Path,
        action: dict[str, Any],
        target_class_label: str,
    ) -> bool:
        """Check if there are multimple instances of the target object across frames."""
        ambiguous_across_frames = False
        for other_frame_index, _ in enumerate(action["colorImages"]):
            if frame_index == other_frame_index:
                continue
            candidate_objects = self.object_decoder.get_candidate_object_in_frame(
                features_path=features_path,
                frame_index=other_frame_index,
                target_class_label=target_class_label,
            )
            if candidate_objects:
                ambiguous_across_frames = True
                break
        return ambiguous_across_frames

    def ambiguous_in_frame(
        self,
        frame_index: int,
        features_path: Path,
        target_class_label: str,
    ) -> bool:
        """Check if there are multimple instances of the target object in the frame."""
        candidate_objects = self.object_decoder.get_candidate_object_in_frame(
            features_path=features_path,
            frame_index=frame_index,
            target_class_label=target_class_label,
        )
        if len(candidate_objects) > 1:
            return self.check_no_salient_object(
                features_path=features_path,
                frame_index=frame_index,
                candidate_objects=candidate_objects,
                target_class_label=target_class_label,
            )
        return False

    def not_in_frame(self, frame_index: int, features_path: Path, target_class_label: str) -> bool:
        """Check if there is no target object in the frame."""
        candidate_objects = self.object_decoder.get_candidate_object_in_frame(
            frame_index=frame_index,
            features_path=features_path,
            target_class_label=target_class_label,
        )
        return len(candidate_objects) == 0  # noqa: WPS507

    def target_same_as_readable_name(self, action: SimBotAction, target_class_label: str) -> bool:
        """Check if the target and readable names are the same."""
        _, _, readable_name = self.object_decoder.get_target_object_and_name(action.dict())
        return target_class_label == readable_name

    def holding_object(self, action: SimBotAction, target_class_label: str) -> bool:
        """Check if the agent is holding the target object."""
        if target_class_label == "Slice":
            return action.holding_object in {"Apple", "Pie", "Cake", "Bread"}
        return action.holding_object == target_class_label

    def check_no_salient_object(
        self,
        features_path: Path,
        frame_index: int,
        candidate_objects: list[int],
        target_class_label: str,
    ) -> bool:
        """No salient object means that the instruction is still ambiguous."""
        features = self.object_decoder.load_features(features_path, frame_index=frame_index)
        if not features:
            return True
        # Filter small bboxes based on area
        candidate_bboxes = self._filter_bboxes_based_on_area(
            [features["bbox_coords"][idx] for idx in candidate_objects],
            target_class_label=target_class_label,
        )
        if len(candidate_bboxes) == 1:
            return False
        # Now try to determine saliency
        candidate_areas = [compute_bbox_area(bbox) for bbox in candidate_bboxes]
        candidate_xcenter = [compute_bbox_center_coords(bbox)[0] for bbox in candidate_bboxes]
        no_salient_bbox = True
        for index, (area, xcenter) in enumerate(zip(candidate_areas, candidate_xcenter)):
            # An object is salient if it's centered in the image
            cond1 = self._min_center_coord <= xcenter <= self._max_center_coord
            if cond1:
                # An object is salient if it is relatively large compared to other candidate objects
                area_comparison = [
                    area >= self._area_percentage * other_area for other_area in candidate_areas
                ]
            else:
                # The area is much bigger than other candidates
                area_comparison = [
                    area >= 3 * other_area
                    for other_index, other_area in enumerate(candidate_areas)
                    if other_index != index
                ]
            cond2 = all(area_comparison)
            if cond2:
                return False
        return no_salient_bbox

    def _filter_bboxes_based_on_area(
        self,
        candidate_bboxes: list[torch.Tensor],
        target_class_label: str,
    ) -> list[torch.Tensor]:
        """Return relatively large."""
        filtered_bboxes = []
        thresholds = self._class_thresholds.get(target_class_label, None)
        if thresholds is None:
            threshold = self._default_min_area
        else:
            threshold = min(thresholds[0] * 5, self._default_min_area)
        for bbox in candidate_bboxes:
            if compute_bbox_area(bbox) > threshold:
                filtered_bboxes.append(bbox)
        return filtered_bboxes


class AmbiguousGotoProcessor(AmbiguityProcessor):
    """Preprocess ambiguous low-level Goto-object instructions.

    An instruction is considered ambiguous if more than one objects of the object class have been
    detected. If multiple objects are present in the same frame, we try to make the instruction
    unambiguous by using attributes associated with the asset name.
    """

    def __init__(self) -> None:
        super().__init__()
        self.object_decoder = ObjectClassDecoder()

    def __call__(
        self, instruction_dict: dict[str, Any], mission_id: str, action: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Annotate an instruction."""
        if action["type"] != "Goto" or "officeRoom" in action["goto"]["object"]:
            return [instruction_dict]
        (
            target_object,
            target_class_label,
            target_readable_name,
        ) = self.object_decoder.get_target_object_and_name(action)

        frame_index = action["goto"]["object"]["colorImageIndex"]

        new_intruction_dict = None
        action_id = action["id"]
        features_path = settings.paths.simbot_features.joinpath(
            f"{mission_id}_action{action_id}.pt"
        )
        ambiguous_in_frame = self.ambiguous_in_frame(
            frame_index=frame_index,
            features_path=features_path,
            target_class_label=target_class_label,
        )
        if ambiguous_in_frame:
            new_intruction_dict = self._try_to_fix_ambiguous_in_frame(
                instruction_dict=instruction_dict,
                target_object=target_object,
                target_readable_name=target_readable_name,
            )
            instruction_dict["ambiguous"] = True

        if new_intruction_dict is not None:
            return [instruction_dict, new_intruction_dict]

        return [instruction_dict]

    def _try_to_fix_ambiguous_in_frame(
        self, instruction_dict: dict[str, Any], target_object: str, target_readable_name: str
    ) -> Optional[dict[str, Any]]:
        """Use attributes from the target object asset name to make the instruction unambiguous."""
        if target_object != target_readable_name:
            return None

        new_instruction_dict = deepcopy(instruction_dict)
        new_instruction_dict["instruction"][
            "instruction"
        ] = f"go to the {target_readable_name.lower()}"
        new_instruction_dict["ambiguous"] = False
        new_instruction_dict[
            "annotation_id"
        ] = f"{new_instruction_dict['annotation_id']}_one_match"
        return new_instruction_dict


class ClarificationFilter:
    """Filter clarification questions.

    Keep only:
    1. disambiguation questions when there is ambiguity in the front frame.
    2. location questions when the target object is not in the front frame.
    """

    def __init__(self) -> None:
        self.ambiguity_processor = AmbiguityProcessor()
        self.clarification_target_extractor = ClarificationTargetExtractor()
        self._skip_instruction_word_list = [
            "locate",
            "find",
            "search",
            "look for",
            "trace",
            "seek",
        ]
        self._disambiguation_keyword_list = ["blue", "red", "green", "with"]
        self._assets_to_labels = get_arena_definitions()["asset_to_label"]

    def __call__(
        self, instruction_instance: SimBotInstructionInstance
    ) -> Optional[list[SimBotQA]]:
        """Filter the questions."""
        question_answers = instruction_instance.instruction.necessary_question_answers
        if not question_answers:
            return None
        new_question_answers = []
        for qa_pair in question_answers:
            keep_qa_pair = self._keep_qa_pair(
                qa_pair=qa_pair,
                features_path=instruction_instance.features_path[0],
                action=instruction_instance.actions[0],
                instruction=instruction_instance.instruction.instruction,
            )
            if keep_qa_pair:
                new_question_answers.append(qa_pair)

        if not new_question_answers:
            return None
        return new_question_answers

    def skip_instruction(self, instruction: str) -> bool:
        """Skip human instructions that can be confused with Search instructions."""
        instruction = instruction.lower()
        for skip_word in self._skip_instruction_word_list:
            if skip_word in instruction:
                return True

        return False

    def _keep_qa_pair(
        self,
        qa_pair: SimBotQA,
        action: SimBotAction,
        instruction: str,
        features_path: Path,
    ) -> bool:
        keep_qa_pair = False
        # Fist, check the question type
        if not self._keeping_question_type(qa_pair):
            return keep_qa_pair

        # Convert the question target to an object detector label
        target_object = self.clarification_target_extractor.normalize_target(
            qa_pair.question_target, instruction
        )
        if target_object is None:
            return True
        # Second, check if the agent is holding the target object
        holding_clarification_target = self.ambiguity_processor.holding_object(
            action, target_class_label=target_object
        )
        if holding_clarification_target:
            return keep_qa_pair

        # Finally, check conditions based on the question type
        if qa_pair.question_type == SimBotClarificationTypes.disambiguation:
            keep_qa_pair = self._filter_disambiguation_questions(
                features_path=features_path,
                action=action,
                instruction=instruction,
                target_object=target_object,
            )
        elif qa_pair.question_type == SimBotClarificationTypes.location:
            keep_qa_pair = self._filter_location_questions(
                features_path=features_path,
                action=action,
                instruction=instruction,
                target_object=target_object,
            )

        if qa_pair.question_target != "desk":
            qa_pair.question_target = target_object.lower()
        return keep_qa_pair

    def _keeping_question_type(self, qa_pair: SimBotQA) -> bool:
        return qa_pair.question_type in {
            SimBotClarificationTypes.location,
            SimBotClarificationTypes.disambiguation,
        }

    def _first_target_is_unique(self, action: SimBotAction, features_path: Path) -> bool:
        """Skip instances when there is one instance matching the target."""
        target_object_type = action.get_action_data["object"]["id"]
        target_object = get_object_label_from_object_id(target_object_type, self._assets_to_labels)
        ambiguous_target = self.ambiguity_processor.ambiguous_in_frame(
            frame_index=action.get_action_data["object"]["colorImageIndex"],
            features_path=features_path,
            target_class_label=target_object,
        )
        return not ambiguous_target

    def _check_instruction_keywords(self, instruction: str) -> bool:
        """Check for keywords that make the instruction probably not ambiguous."""
        for keyword in self._disambiguation_keyword_list:
            if keyword in instruction:
                return True
        return False

    def _filter_disambiguation_questions(
        self, features_path: Path, action: SimBotAction, instruction: str, target_object: str
    ) -> bool:
        """Filter disambiguation questions."""
        keyword_exists = self._check_instruction_keywords(instruction)
        if keyword_exists:
            return False
        if self._first_target_is_unique(action, features_path):
            return False
        return self.ambiguity_processor.ambiguous_in_frame(
            frame_index=action.get_action_data["object"]["colorImageIndex"],
            features_path=features_path,
            target_class_label=target_object,
        )

    def _filter_location_questions(
        self, features_path: Path, action: SimBotAction, instruction: str, target_object: str
    ) -> bool:
        """Filter location questions."""
        target_same_as_readable_name = self.ambiguity_processor.target_same_as_readable_name(
            action=action,
            target_class_label=target_object,
        )
        target_not_in_frame = self.ambiguity_processor.not_in_frame(
            frame_index=0,
            features_path=features_path,
            target_class_label=target_object,
        )
        return target_same_as_readable_name and target_not_in_frame


class VisionAugmentationFilter:
    """Filter vision augmentation instances.

    Remove ambiguous instructions.
    """

    def __init__(self) -> None:
        self.ambiguity_processor = AmbiguityProcessor()
        arena_definitions = get_arena_definitions()
        self._assets_to_labels = arena_definitions["asset_to_label"]
        self._special_names = arena_definitions["special_asset_to_readable_name"]

    def __call__(
        self,
        instruction_instance: SimBotInstructionInstance,
    ) -> bool:
        """Filter the vision augmentation data."""
        action = instruction_instance.actions[0]
        if action.type == "Search":
            search_object_metadata = action.search.get("selected_object", None)
            if search_object_metadata is None:
                return False
            target_object_id = search_object_metadata["id"]
        else:
            target_object_id = action.get_action_data["object"]["id"]
        # Get the readable name of the object
        target_object = get_object_readable_name_from_object_id(
            target_object_id, self._assets_to_labels, self._special_names
        )

        return self.ambiguity_processor.ambiguous_in_frame(
            features_path=instruction_instance.features_path[0],
            frame_index=action.get_action_data["object"]["colorImageIndex"],
            target_class_label=target_object,
        )
