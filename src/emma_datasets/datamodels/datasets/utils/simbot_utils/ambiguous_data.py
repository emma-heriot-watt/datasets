from typing import Any, Optional

import torch

from emma_datasets.constants.simbot.simbot import get_arena_definitions, get_class_thresholds
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    ClarificationTargetExtractor,
    get_object_label_from_object_id,
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
        self, frame_index: int, mission_id: str, action: dict[str, Any], target_object_name: str
    ) -> bool:
        """Check if there are multimple instances of the target object across frames."""
        ambiguous_across_frames = False
        for other_frame_index, _ in enumerate(action["colorImages"]):
            if frame_index == other_frame_index:
                continue
            candidate_objects = self.object_decoder.get_candidate_object_in_frame(
                mission_id=mission_id,
                action_id=action["id"],
                frame_index=other_frame_index,
                target_object_name=target_object_name,
            )
            if candidate_objects:
                ambiguous_across_frames = True
                break
        return ambiguous_across_frames

    def ambiguous_in_frame(
        self, frame_index: int, mission_id: str, action_id: int, target_object_name: str
    ) -> bool:
        """Check if there are multimple instances of the target object in the frame."""
        candidate_objects = self.object_decoder.get_candidate_object_in_frame(
            mission_id=mission_id,
            action_id=action_id,
            frame_index=frame_index,
            target_object_name=target_object_name,
        )
        if len(candidate_objects) > 1:
            return self.check_no_salient_object(
                mission_id=mission_id,
                action_id=action_id,
                frame_index=frame_index,
                candidate_objects=candidate_objects,
                target_object_name=target_object_name,
            )
        return False

    def not_in_frame(
        self, frame_index: int, mission_id: str, action_id: int, target_object_name: str
    ) -> bool:
        """Check if there is no target object in the frame."""
        candidate_objects = self.object_decoder.get_candidate_object_in_frame(
            mission_id=mission_id,
            action_id=action_id,
            frame_index=frame_index,
            target_object_name=target_object_name,
        )
        return len(candidate_objects) == 0  # noqa: WPS507

    def holding_object(self, action: SimBotAction, target_object_name: str) -> bool:
        """Check if the agent is holding the target object."""
        if target_object_name == "Slice":
            return action.holding_object in {"Apple", "Pie", "Cake", "Bread"}
        return action.holding_object == target_object_name

    def check_no_salient_object(
        self,
        mission_id: str,
        action_id: int,
        frame_index: int,
        candidate_objects: list[int],
        target_object_name: str,
    ) -> bool:
        """No salient object means that the instruction is still ambiguous."""
        features = self.object_decoder.load_features(
            mission_id=mission_id, action_id=action_id, frame_index=frame_index
        )
        if not features:
            return True
        # Filter bboxes based on area
        candidate_bboxes = self._filter_bboxes_based_on_area(
            [features["bbox_coords"][idx] for idx in candidate_objects],
            target_object_name=target_object_name,
        )
        if len(candidate_bboxes) == 1:
            return False
        # Now try to determine saliency
        candidate_areas = [compute_bbox_area(bbox) for bbox in candidate_bboxes]
        candidate_xcenter = [compute_bbox_center_coords(bbox)[0] for bbox in candidate_bboxes]
        no_salient_bbox = True
        for area, xcenter in zip(candidate_areas, candidate_xcenter):
            # An object is salient if it's centered in the image
            cond1 = self._min_center_coord <= xcenter <= self._max_center_coord
            if not cond1:
                continue
            # An object is salient if it is relatively large compared to other candidate objects
            area_comparison = [
                area >= self._area_percentage * other_area for other_area in candidate_areas
            ]
            cond2 = all(area_comparison)
            if cond2:
                return False

        return no_salient_bbox

    def _filter_bboxes_based_on_area(
        self,
        candidate_bboxes: list[torch.Tensor],
        target_object_name: str,
    ) -> list[torch.Tensor]:
        filtered_bboxes = []
        thresholds = self._class_thresholds.get(target_object_name, None)
        if thresholds is None:
            threshold = self._default_min_area
        else:
            threshold = thresholds[0] * 5
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
    ) -> dict[str, Any]:
        """Annotate an instruction."""
        if action["type"] != "Goto" or "officeRoom" in action["goto"]["object"]:
            return instruction_dict
        target_object, target_object_name = self.object_decoder.get_target_object_and_name(action)
        frame_index = action["goto"]["object"]["colorImageIndex"]
        ambiguous_in_frame = self.ambiguous_in_frame(
            frame_index=frame_index,
            mission_id=mission_id,
            action_id=action["id"],
            target_object_name=target_object_name,
        )
        if ambiguous_in_frame:
            return self._try_to_fix_ambiguous_in_frame(
                instruction_dict=instruction_dict, target_object=target_object
            )

        ambiguous_across_frames = self.ambiguous_across_frames(
            mission_id=mission_id,
            frame_index=frame_index,
            action=action,
            target_object_name=target_object_name,
        )
        if ambiguous_across_frames:
            return self._try_to_fix_ambiguous_across_frames(
                action=action,
                instruction_dict=instruction_dict,
            )

        return instruction_dict

    def _try_to_fix_ambiguous_across_frames(
        self,
        action: dict[str, Any],
        instruction_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep only the frame with the target object."""
        instruction_dict["actions"][-1]["keep_only_current_frame"] = True

        return instruction_dict

    def _try_to_fix_ambiguous_in_frame(
        self, instruction_dict: dict[str, Any], target_object: str
    ) -> dict[str, Any]:
        """Use attributes from the target object asset name to make the instruction unambiguous."""
        if target_object.startswith("AP_Prop_Desk_"):
            color = target_object.split("AP_Prop_Desk_")[-1].split("_")[0].lower()
            new_instruction = f"go to the {color} desk"
        elif target_object.startswith("TableRound"):
            new_instruction = "go to the round table"
        else:
            new_instruction = ""
        if new_instruction:
            instruction_dict["instruction"]["instruction"] = new_instruction
        else:
            instruction_dict["ambiguous"] = True
        return instruction_dict


class ClarificationFilter:
    """Filter clarification questions.

    Keep only:
    1. disambiguation questions when there is ambiguity in the front frame.
    2. location questions when the target object is not in the front frame.
    """

    def __init__(self) -> None:
        self.ambiguity_processor = AmbiguityProcessor()
        self.clarification_target_extractor = ClarificationTargetExtractor()
        self._skip_instruction_word_list = ["locate", "find", "search"]
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
                mission_id=instruction_instance.mission_id,
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
        self, qa_pair: SimBotQA, mission_id: str, action: SimBotAction, instruction: str
    ) -> bool:
        keep_qa_pair = False
        # Fist, check the question type
        if not self._keeping_question_type(qa_pair):
            return keep_qa_pair

        # Convert the question target to an object detector label
        target_object = self.clarification_target_extractor.normalize_target(
            qa_pair.question_target, instruction
        )
        # Second, check if the agent is holding the target object
        holding_clarification_target = self.ambiguity_processor.holding_object(
            action, target_object_name=target_object
        )
        if holding_clarification_target:
            return keep_qa_pair

        # Finally, check conditions based on the question type
        if qa_pair.question_type == SimBotClarificationTypes.disambiguation:
            keyword_exists = self._check_instruction_keywords(instruction)
            if keyword_exists:
                return keep_qa_pair
            if self._first_target_is_unique(action, mission_id):
                return keep_qa_pair
            keep_qa_pair = self.ambiguity_processor.ambiguous_in_frame(
                frame_index=action.get_action_data["object"]["colorImageIndex"],
                mission_id=mission_id,
                action_id=action.id,
                target_object_name=target_object,
            )
        elif qa_pair.question_type == SimBotClarificationTypes.location:
            keep_qa_pair = self.ambiguity_processor.not_in_frame(
                frame_index=0,
                mission_id=mission_id,
                action_id=action.id,
                target_object_name=target_object,
            )
        if qa_pair.question_target != "desk":
            qa_pair.question_target = target_object.lower()
        return keep_qa_pair

    def _keeping_question_type(self, qa_pair: SimBotQA) -> bool:
        return qa_pair.question_type in {
            SimBotClarificationTypes.location,
            SimBotClarificationTypes.disambiguation,
        }

    def _first_target_is_unique(self, action: SimBotAction, mission_id: str) -> bool:
        """Skip instances when there is one instance matching the target."""
        target_object_type = action.get_action_data["object"]["id"]
        target_object = get_object_label_from_object_id(target_object_type, self._assets_to_labels)
        ambiguous_target = self.ambiguity_processor.ambiguous_in_frame(
            frame_index=action.get_action_data["object"]["colorImageIndex"],
            mission_id=mission_id,
            action_id=action.id,
            target_object_name=target_object,
        )
        return not ambiguous_target

    def _check_instruction_keywords(self, instruction: str) -> bool:
        """Check for keywords that make the instruction probably not ambiguous."""
        for keyword in self._disambiguation_keyword_list:
            if keyword in instruction:
                return True
        return False
