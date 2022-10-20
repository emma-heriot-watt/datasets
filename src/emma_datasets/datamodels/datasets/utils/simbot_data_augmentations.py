import random
from typing import Any, Optional

import torch

from emma_datasets.common import Settings
from emma_datasets.constants.simbot.simbot import (
    get_arena_definitions,
    get_low_level_action_templates,
)
from emma_datasets.datamodels.datasets.utils.masks import compress_simbot_mask
from emma_datasets.datamodels.datasets.utils.simbot_utils import (
    get_object_from_action_object_metadata,
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

    def __init__(self) -> None:
        arena_definitions = get_arena_definitions()

        self.idx_to_label = {
            idx: label for label, idx in arena_definitions["label_to_idx"].items()
        }
        self.object_assets_to_names = arena_definitions["asset_to_name"]
        self._annotation_id = "synthetic_goto"

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
        target_object, target_object_name = self._get_target_object_and_name(
            instruction_actions[2]
        )
        # Prepare the new Goto instruction
        synthetic_instruction = {
            "instruction": f"go to the {target_object_name.lower()}",
            "actions": [instruction_actions[1]["id"]],
        }
        # Get a mask for the target object
        mask = self._get_target_object_mask(
            mission_id=mission_id,
            instruction_actions=instruction_actions,
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
        goto_object = self._get_target_object(instruction_actions[1])
        target_object = self._get_target_object(instruction_actions[2])
        return goto_object != target_object

    def _get_target_object(self, action: dict[str, Any]) -> str:
        """Get the target object id for an action."""
        action_type = action["type"].lower()
        return action[action_type]["object"]["id"]

    def _get_target_object_and_name(self, action: dict[str, Any]) -> tuple[str, str]:
        """Get the target object id and name for an action."""
        target_object = self._get_target_object(action)
        target_object_name = get_object_from_action_object_metadata(
            target_object, self.object_assets_to_names
        )
        return target_object, target_object_name

    def _get_target_object_mask(
        self, mission_id: int, instruction_actions: list[dict[str, Any]], target_object_name: str
    ) -> Optional[list[list[int]]]:
        # Load the features from the Goto action
        action_id = instruction_actions[1]["id"]
        features_path = settings.paths.simbot_features.joinpath(
            f"{mission_id}_action{action_id}.pt"
        )
        image_index = instruction_actions[1]["goto"]["object"]["colorImageIndex"]
        features = torch.load(features_path)["frames"][image_index]["features"]
        # Get the class indices for the predicted boxes
        class_indices = torch.argmax(features["bbox_probas"], dim=1).tolist()
        # Get the indices of the objects that match the target_object_name
        candidate_objects = [
            idx
            for idx, class_idx in enumerate(class_indices)
            if self.idx_to_label[class_idx] == target_object_name
        ]
        if not candidate_objects:
            return None
        # Keep the bounding box for one matching object
        (x_min, y_min, x_max, y_max) = features["bbox_coords"][candidate_objects[0]].tolist()
        # Convert bbox to mask
        mask = torch.zeros((features["width"], features["height"]))
        # populate the bbox region in the mask with ones
        mask[int(y_min) : int(y_max) + 1, int(x_min) : int(x_max) + 1] = 1  # noqa: WPS221
        compressed_mask = compress_simbot_mask(mask)
        return compressed_mask
