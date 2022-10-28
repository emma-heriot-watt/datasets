import random
from typing import Any, Optional

from emma_datasets.common.settings import Settings
from emma_datasets.constants.simbot.simbot import get_low_level_action_templates
from emma_datasets.datamodels.datasets.utils.simbot.object_features_processing import (
    ObjectClassDecoder,
)


settings = Settings()


class SyntheticLowLevelActionSampler:
    """Create synthetic examples of low level actions."""

    def __init__(self) -> None:

        self.examine_action = "Examine"
        self._low_level_action_templates = get_low_level_action_templates()
        # TODO: Examine for now is only reserved for the sticky notes
        # but apparently the examine can be used for any other object as well
        # https://alexaprizesim-ldg5293.slack.com/files/U02SFPST8AK/F043B3MAX1S/arena_for_embodied_ai_-_user_manual.pdf
        self._low_level_actions = [
            key for key in self._low_level_action_templates.keys() if key != self.examine_action
        ]

    def __call__(
        self,
        mission_id: int,
        annotation_id: str,
        instruction_idx: int,
        original_action: Optional[dict[str, Any]] = None,
        sample_sticky_note: bool = False,
        sticky_note_image: Optional[str] = None,
        sticky_note_bbox_coords: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Sample a low level action and an instruction template."""
        if sample_sticky_note:
            if sticky_note_image is None:  # or sticky_note_image_layout is None:
                raise AssertionError("Need a path to a sticky note image and the image layout")

            low_level_action = self.examine_action
            low_level_action_template = random.choice(
                self._low_level_action_templates[low_level_action]["templates"]
            )

            action_type = self._low_level_action_templates[low_level_action]["type"]
            action_id = 0
            color_images = [sticky_note_image]
            payload = {
                "object": {
                    "id": "Sticky Note",
                    "mask": sticky_note_bbox_coords,
                    "colorImageIndex": 0,
                }
            }
        else:
            if original_action is None:
                raise AssertionError("Need the original actions")
            low_level_action = random.choice(self._low_level_actions)
            low_level_action_template = random.choice(
                self._low_level_action_templates[low_level_action]["templates"]
            )
            action_type = self._low_level_action_templates[low_level_action]["type"]
            action_id = original_action["id"]
            color_images = original_action["colorImages"]
            payload = {
                "direction": self._low_level_action_templates[low_level_action]["direction"]
            }

        synthetic_instruction = {
            "instruction": low_level_action_template,
            "actions": [action_id],
        }
        synthetic_action = {
            "id": action_id,
            "type": action_type,
            action_type.lower(): payload,
            "colorImages": color_images,
            "final": True,
        }
        instruction_dict = {
            "instruction": synthetic_instruction,
            "actions": [synthetic_action],
            "mission_id": mission_id,
            "annotation_id": annotation_id,
            "instruction_id": str(instruction_idx),
            "synthetic": True,
        }
        return instruction_dict


class SyntheticGotoObjectGenerator:
    """Create synthetic examples of go to actions for interactable objects."""

    def __init__(
        self,
    ) -> None:

        self._annotation_id = "synthetic_goto"
        self.object_decoder = ObjectClassDecoder()

    def __call__(
        self,
        mission_id: int,
        instruction_idx: int,
        instruction_actions: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """Generate a new Goto action."""
        synthetic_instruction = None
        synthetic_action = None
        if not self._check_usable_instruction(instruction_actions):
            return None

        # Get the target object from the 3rd action
        target_object, target_object_name = self.object_decoder.get_target_object_and_name(
            instruction_actions[2]
        )
        # Prepare the new Goto instruction
        synthetic_instruction = {
            "instruction": f"go to the {target_object_name.lower()}",
            "actions": [instruction_actions[1]["id"]],
        }
        # Get a mask for the target object
        mask = self.object_decoder.get_target_object_mask(
            mission_id=mission_id,
            action_id=instruction_actions[1]["id"],
            frame_index=instruction_actions[1]["goto"]["object"]["colorImageIndex"],
            target_object_name=target_object_name,
        )
        if not mask:
            return None
        synthetic_action = {
            "id": instruction_actions[1]["id"],
            "type": "Goto",
            "goto": {
                "object": {
                    "id": target_object,
                    "colorImageIndex": instruction_actions[1]["goto"]["object"]["colorImageIndex"],
                    "mask": mask,
                }
            },
            "colorImages": instruction_actions[1]["colorImages"],
            "final": True,
        }

        instruction_dict = {
            "instruction": synthetic_instruction,
            "actions": [synthetic_action],
            "mission_id": mission_id,
            "annotation_id": self._annotation_id,
            "instruction_id": str(instruction_idx),
            "synthetic": True,
        }
        return instruction_dict

    def _check_usable_instruction(
        self,
        instruction_actions: list[dict[str, Any]],
    ) -> bool:
        """Check if the given instruction can be converted into a new Goto instruction.

        1. Instructions should follow the pattern (Look Around, Goto <object 1>, Interact <object 2>)
        2. <object 2> should not be a receptacle, i.e. Interact should not be "place"
        3. <object 1> should be different from <object 2>
        """
        if len(instruction_actions) != 3:
            return False
        action_pattern = (
            instruction_actions[0]["type"] == "Look" and instruction_actions[1]["type"] == "Goto"
        )
        if not action_pattern:
            return False
        goto_object = self.object_decoder.get_target_object(instruction_actions[1])
        target_object = self.object_decoder.get_target_object(instruction_actions[2])
        return goto_object != target_object
