from typing import Any

from emma_datasets.datamodels.datasets.utils.simbot_utils.object_features_processing import (
    ObjectClassDecoder,
)


class AmbiguousGotoProcessor:
    """Preprocess ambiguous low-level Goto-object instructions.

    An instruction is considered ambiguous if more than one objects of the object class have been
    detected. If multiple objects are present in the same frame, we try to make the instruction
    unambiguous by using attributes associated with the asset name.
    """

    def __init__(self) -> None:
        self.object_decoder = ObjectClassDecoder()

    def __call__(
        self, instruction_dict: dict[str, Any], mission_id: str, action: dict[str, Any]
    ) -> dict[str, Any]:
        """Annotate an instruction."""
        if action["type"] != "Goto" or "officeRoom" in action["goto"]["object"]:
            return instruction_dict
        target_object, target_object_name = self.object_decoder.get_target_object_and_name(action)
        frame_index = action["goto"]["object"]["colorImageIndex"]
        ambiguous_in_frame = self._ambiguous_in_frame(
            frame_index=frame_index,
            mission_id=mission_id,
            action_id=action["id"],
            target_object_name=target_object_name,
        )
        if ambiguous_in_frame:
            return self._try_to_fix_ambiguous_in_frame(
                instruction_dict=instruction_dict, target_object=target_object
            )

        ambiguous_across_frames = self._ambiguous_across_frames(
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

    def _ambiguous_across_frames(
        self, frame_index: int, mission_id: str, action: dict[str, Any], target_object_name: str
    ) -> bool:
        ambiguous_across_frames = False
        for other_frame_index, _ in enumerate(action["colorImages"]):
            if frame_index == other_frame_index:
                continue
            ambiguous_across_frames = self._ambiguous_in_frame(
                frame_index=other_frame_index,
                mission_id=mission_id,
                action_id=action["id"],
                target_object_name=target_object_name,
            )
            if ambiguous_across_frames:
                break
        return ambiguous_across_frames

    def _ambiguous_in_frame(
        self, frame_index: int, mission_id: str, action_id: int, target_object_name: str
    ) -> bool:
        candidate_objects = self.object_decoder.get_candidate_object_in_frame(
            mission_id=mission_id,
            action_id=action_id,
            frame_index=frame_index,
            target_object_name=target_object_name,
        )
        return len(candidate_objects) > 1

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
