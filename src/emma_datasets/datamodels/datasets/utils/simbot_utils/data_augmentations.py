import random
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from emma_datasets.common.settings import Settings
from emma_datasets.constants.simbot.simbot import (
    get_arena_definitions,
    get_low_level_action_templates,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.object_features_processing import (
    ObjectClassDecoder,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    AugmentationInstruction,
    SimBotObjectAttributes,
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
        mission_id: str,
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
                    "mask": [sticky_note_bbox_coords],
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
        mission_id: str,
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


class BaseAugmentation:
    """Base class for object augmentations."""

    def __init__(self) -> None:
        self._assets_to_labels = get_arena_definitions()["asset_to_label"]

    def __call__(
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
    ) -> list[AugmentationInstruction]:
        """Creates new annotations for a given object."""
        raise NotImplementedError("Do not call BaseAugmentation class")

    def _compute_bbox_center(self, bbox: list[int]) -> tuple[float, float]:
        (x_min, y_min, x_max, y_max) = bbox
        return ((x_max - x_min) / 2, (y_max - y_min) / 2)

    def _compute_bbox_area(self, bbox: list[int]) -> float:
        return (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])


class SpecialMonitorAugmentation(BaseAugmentation):
    """Monitor Augmentations."""

    def __init__(self, min_interaction_distance: float = 1.5) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self._monitor_color_map = {
            "Freeze ray monitor": "blue",
            "Gravity flipper monitor": "green",
            "Laser monitor": "red",
            "Embiggenator monitor": "pink",
        }
        self._monitor_object_type_map = {
            "V_Monitor_FreezeRay": "Freeze ray monitor",
            "V_Monitor_Gravity": "Gravity flipper monitor",
            "V_Monitor_Laser": "Laser monitor",
            "V_Monitor_Embiggenator": "Embiggenator monitor",
        }

    def __call__(  # noqa: WPS231
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
    ) -> list[AugmentationInstruction]:
        """Get new annotations for monitors."""
        interaction_instructions = []
        navigation_instructions = []
        for _, annotation in annotations.items():
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]
            bbox = image_annotation["bbox"]
            readable_name = self._monitor_object_type_map.get(object_type, None)
            if readable_name is None:
                continue
            # all special monitors have Computer as class label
            # ignore the maximum threshold value
            if self._compute_bbox_area(bbox) < class_thresholds["Computer"][0]:
                continue
            color_name = self._monitor_color_map[readable_name]

            object_position = np.array(
                [
                    object_annotation["position"]["x"],
                    object_annotation["position"]["y"],
                    object_annotation["position"]["z"],
                ]
            )
            distance2object = np.linalg.norm(robot_position - object_position)

            # Monitor is within reach - interaction instruction
            if distance2object <= self.min_interaction_distance:
                instruction = AugmentationInstruction(
                    action_type="Toggle",
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name, color=color_name
                    ),
                    bbox=bbox,
                    image_name=image_name,
                )
                interaction_instructions.append(instruction)

            # Monitor is out of reach - navigation instruction
            else:
                instruction = AugmentationInstruction(
                    action_type="Goto",
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name, color=color_name
                    ),
                    bbox=bbox,
                    image_name=image_name,
                )
                navigation_instructions.append(instruction)

        # Image has multiple monitors, include instructions with spatial or color information
        if len(interaction_instructions) > 1:
            interaction_instructions.extend(
                self._get_instructions_from_attributes(interaction_instructions)
            )

        if len(navigation_instructions) > 1:
            navigation_instructions.extend(
                self._get_instructions_from_attributes(navigation_instructions)
            )

        return interaction_instructions + navigation_instructions

    def _get_instructions_from_attributes(
        self, instruction_list: list[AugmentationInstruction]
    ) -> list[AugmentationInstruction]:
        bbox_centers = [
            self._compute_bbox_center(instruction.bbox) for instruction in instruction_list
        ]

        left2right = np.argsort([bbox_center[0] for bbox_center in bbox_centers])
        left_instruction = AugmentationInstruction(
            action_type=instruction_list[left2right[0]].action_type,
            object_id=instruction_list[left2right[0]].object_id,
            attributes=SimBotObjectAttributes(
                readable_name=instruction_list[left2right[0]].attributes.readable_name,
                color=instruction_list[left2right[0]].attributes.color,
                location="left",
            ),
            bbox=instruction_list[left2right[0]].bbox,
            image_name=instruction_list[left2right[0]].image_name,
        )

        right_instruction = AugmentationInstruction(
            action_type=instruction_list[left2right[-1]].action_type,
            object_id=instruction_list[left2right[-1]].object_id,
            attributes=SimBotObjectAttributes(
                readable_name=instruction_list[left2right[-1]].attributes.readable_name,
                color=instruction_list[left2right[-1]].attributes.color,
                location="right",
            ),
            bbox=instruction_list[left2right[-1]].bbox,
            image_name=instruction_list[left2right[-1]].image_name,
        )
        return [left_instruction, right_instruction]
