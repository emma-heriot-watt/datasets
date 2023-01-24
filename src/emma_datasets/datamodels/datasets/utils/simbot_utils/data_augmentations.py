import random
from collections import defaultdict
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from emma_datasets.common.settings import Settings
from emma_datasets.constants.simbot.simbot import (
    get_arena_definitions,
    get_low_level_action_templates,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_asset_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.object_features_processing import (
    ObjectClassDecoder,
    compute_bbox_area,
    compute_bbox_center_coords,
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
        target_object, target_object_name, _ = self.object_decoder.get_target_object_and_name(
            instruction_actions[2]
        )

        # Prepare the new Goto instruction
        action_id = instruction_actions[1]["id"]
        synthetic_instruction = {
            "instruction": f"go to the {target_object_name.lower()}",
            "actions": [action_id],
        }
        # Get a mask for the target object
        mask = self.object_decoder.get_target_object_mask(
            mission_id=mission_id,
            action_id=action_id,
            frame_index=instruction_actions[1]["goto"]["object"]["colorImageIndex"],
            target_class_label=target_object_name,
        )
        if not mask:
            return None
        synthetic_action = {
            "id": action_id,
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
        label_to_index = get_arena_definitions()["label_to_idx"]
        self._index_to_label = {index: label for label, index in label_to_index.items()}

        self._object_color_map = {
            "Embiggenator Monitor": "pink",
            "Freeze Ray Monitor": "blue",
            "Freeze Ray Shelf": "blue",
            "Gravity Monitor": "green",
            "Laser Monitor": "red",
            "Laser Shelf": "red",
        }

        self._special_object_type_map = {
            "AP_Prop_Shelf_Wall_04": "Freeze Ray Shelf",
            "AP_Prop_Shelf_Wall_FreezeRay": "Freeze Ray Shelf",
            "AP_Prop_Shelf_Wall_Laser": "Laser Shelf",
            "V_Monitor_Embiggenator": "Embiggenator Monitor",
            "V_Monitor_FreezeRay": "Freeze Ray Monitor",
            "V_Monitor_Gravity": "Gravity Monitor",
            "V_Monitor_Laser": "Laser Monitor",
        }

    def __call__(
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Creates new annotations for a given object."""
        raise NotImplementedError("Do not call BaseAugmentation class")

    def post_process_metadata(self, action_type_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post-process any annotation."""
        return action_type_metadata

    def _compute_distance_to_object(
        self, object_annotation: dict[str, Any], robot_position: NDArray[np.float32]
    ) -> np.float32:
        object_position = np.array(
            [
                object_annotation["position"]["x"],
                object_annotation["position"]["y"],
                object_annotation["position"]["z"],
            ]
        )
        distance_to_object = np.linalg.norm(robot_position - object_position)
        return distance_to_object

    def _get_instructions_from_attributes(
        self,
        instruction_list: list[AugmentationInstruction],
        annotation_id: int,
    ) -> tuple[list[AugmentationInstruction], int]:
        instructions = []
        bbox_centers = [
            compute_bbox_center_coords(instruction.bbox) for instruction in instruction_list  # type: ignore[arg-type]
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
            annotation_id=annotation_id,
        )
        instructions.append(left_instruction)
        annotation_id += 1

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
            annotation_id=annotation_id,
        )

        instructions.append(right_instruction)
        annotation_id += 1

        return instructions, annotation_id


class GoToAugmentation(BaseAugmentation):
    """Goto Augmentations."""

    def __init__(
        self,
        goto_classes: list[str],
        min_interaction_distance: float = 2.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class

        # Force goto special monitors
        self.goto_objects = goto_classes + list(self._special_object_type_map.values())

    def __call__(  # noqa: WPS231
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Get new annotations for the selected classes."""
        navigation_instructions_dict = defaultdict(list)
        annotation_id = 0
        for _, annotation in annotations.items():
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]
            if object_type == "Unassigned":
                continue

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            # Ignore objects that are not specified
            if readable_name not in self.goto_objects:
                continue

            # Ignore too small objects
            bbox = image_annotation["bbox"]
            if compute_bbox_area(bbox) < class_thresholds[object_class][0]:
                continue

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )
            if distance_to_object > self.min_interaction_distance:
                instruction = AugmentationInstruction(
                    action_type="Goto",
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name,
                        color=self._object_color_map.get(readable_name, None),
                    ),
                    bbox=image_annotation["bbox"],
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )
                annotation_id += 1
                navigation_instructions_dict[object_class].append(instruction)

        navigation_instructions = []
        for _, object_instructions in navigation_instructions_dict.items():
            if len(object_instructions) > 1:
                instructions, annotation_id = self._get_instructions_from_attributes(
                    object_instructions, annotation_id
                )
                navigation_instructions.extend(instructions)
            else:
                navigation_instructions.extend(object_instructions)

        return navigation_instructions

    def post_process_metadata(self, goto_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the search actions.

        This basically downsamples the negative examples in the dataset using a fixed maximum
        number of negative examples per room.
        """
        final_metadata: dict[str, Any] = {}
        goto_metadata_grouped_by_class: dict[str, Any] = {}

        for key, annotation in goto_metadata.items():
            object_id = annotation["actions"][0]["goto"]["object"]["id"]  # noqa: WPS219

            object_asset = get_object_asset_from_object_id(object_id, self._assets_to_labels)
            object_class = self._special_object_type_map.get(
                object_asset, self._assets_to_labels[object_asset]
            )

            instructions = goto_metadata_grouped_by_class.get(object_class, [])
            instructions.append({key: annotation})
            goto_metadata_grouped_by_class[object_class] = instructions

        for _, object_class_metadata in goto_metadata_grouped_by_class.items():
            random.shuffle(object_class_metadata)
            for object_class_annotation in object_class_metadata[: self.max_examples_per_class]:
                final_metadata.update(object_class_annotation)
        return final_metadata


class ToggleAugmentation(BaseAugmentation):
    """Toggle Augmentations."""

    def __init__(
        self,
        toggle_classes: list[str],
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class

        # Force toggle special monitors
        self.toggle_objects = toggle_classes + list(self._special_object_type_map.values())

    def __call__(  # noqa: WPS231
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Get new annotations for the selected classes."""
        toggle_instructions_dict = defaultdict(list)
        annotation_id = 0
        for _, annotation in annotations.items():
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]
            if object_type == "Unassigned":
                continue

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            # Ignore objects that are not specified
            if readable_name not in self.toggle_objects:
                continue

            # Ignore too small objects
            bbox = image_annotation["bbox"]
            if compute_bbox_area(bbox) < class_thresholds[object_class][0]:
                continue

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )
            if distance_to_object <= self.min_interaction_distance:
                instruction = AugmentationInstruction(
                    action_type="Toggle",
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name,
                        color=self._object_color_map.get(readable_name, None),
                    ),
                    bbox=image_annotation["bbox"],
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )
                annotation_id += 1
                toggle_instructions_dict[object_class].append(instruction)

        toggle_instructions = []
        for _, object_instructions in toggle_instructions_dict.items():
            if len(object_instructions) > 1:
                instructions, annotation_id = self._get_instructions_from_attributes(
                    object_instructions, annotation_id
                )
                toggle_instructions.extend(instructions)
            else:
                toggle_instructions.extend(object_instructions)
        return toggle_instructions

    def post_process_metadata(self, toggle_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the search actions.

        This basically downsamples the negative examples in the dataset using a fixed maximum
        number of negative examples per room.
        """
        final_metadata: dict[str, Any] = {}
        toggle_metadata_grouped_by_class: dict[str, Any] = {}

        for key, annotation in toggle_metadata.items():
            object_id = annotation["actions"][0]["toggle"]["object"]["id"]  # noqa: WPS219

            object_asset = get_object_asset_from_object_id(object_id, self._assets_to_labels)
            object_class = self._special_object_type_map.get(
                object_asset, self._assets_to_labels[object_asset]
            )

            instructions = toggle_metadata_grouped_by_class.get(object_class, [])
            instructions.append({key: annotation})
            toggle_metadata_grouped_by_class[object_class] = instructions

        for _, object_class_metadata in toggle_metadata_grouped_by_class.items():
            random.shuffle(object_class_metadata)
            for object_class_annotation in object_class_metadata[: self.max_examples_per_class]:
                final_metadata.update(object_class_annotation)
        return final_metadata


class OpenCloseAugmentation(BaseAugmentation):
    """OpenClose Augmentations."""

    def __init__(
        self,
        action_type_classes: list[str],
        action_type: str = "Open",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class
        self.action_objects = action_type_classes
        self.action_type = action_type

    def __call__(  # noqa: WPS231
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Get new annotations for the selected classes."""
        open_instructions_dict = defaultdict(list)
        annotation_id = 0
        for _, annotation in annotations.items():
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]
            if object_type == "Unassigned":
                continue

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]

            # Ignore objects that are not specified
            if object_class not in self.action_objects:
                continue

            # Ignore too small objects
            bbox = image_annotation["bbox"]
            if compute_bbox_area(bbox) < class_thresholds[object_class][0]:
                continue

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )
            if distance_to_object <= self.min_interaction_distance:
                instruction = AugmentationInstruction(
                    action_type=self.action_type,
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=object_class,
                        color=self._object_color_map.get(object_class, None),
                    ),
                    bbox=image_annotation["bbox"],
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )
                annotation_id += 1
                open_instructions_dict[object_class].append(instruction)

        open_instructions = []
        for _, object_instructions in open_instructions_dict.items():
            if len(object_instructions) > 1:
                instructions, annotation_id = self._get_instructions_from_attributes(
                    object_instructions, annotation_id
                )
                open_instructions.extend(instructions)
            else:
                open_instructions.extend(object_instructions)
        return open_instructions

    def post_process_metadata(self, action_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the search actions.

        This basically downsamples the negative examples in the dataset using a fixed maximum
        number of negative examples per room.
        """
        final_metadata: dict[str, Any] = {}
        action_metadata_grouped_by_class: dict[str, Any] = {}

        for key, annotation in action_metadata.items():
            action_metadata = annotation["actions"][0][self.action_type.lower()]
            object_id = action_metadata["object"]["id"]

            object_asset = get_object_asset_from_object_id(object_id, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]

            instructions = action_metadata_grouped_by_class.get(object_class, [])
            instructions.append({key: annotation})
            action_metadata_grouped_by_class[object_class] = instructions

        for _, object_class_metadata in action_metadata_grouped_by_class.items():
            random.shuffle(object_class_metadata)
            for object_class_annotation in object_class_metadata[: self.max_examples_per_class]:
                final_metadata.update(object_class_annotation)
        return final_metadata


class SearchAugmentation(BaseAugmentation):
    """Search Augmentations."""

    def __init__(
        self,
        search_classes: list[str],
        min_interaction_distance: float = 2.5,
        max_negative_examples_per_room: int = 100,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance

        # Force search special monitors
        self.search_objects = search_classes + list(self._special_object_type_map.values())
        self.max_negative_examples_per_room = max_negative_examples_per_room

    def __call__(  # noqa: WPS231
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Get new annotations for the selected classes."""
        objects_in_image = {search_object: False for search_object in self.search_objects}

        search_object_ids = []
        search_object_bboxes = []
        search_object_attributes = []
        for _, annotation in annotations.items():
            image_annotation = annotation["image_annotation"]
            object_type = image_annotation["object_type"]
            object_annotation = annotation["object_annotation"]
            if object_type == "Unassigned":
                continue

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            # Ignore objects that are not specified
            if readable_name not in self.search_objects:
                continue

            objects_in_image[readable_name] = True

            # Ignore too small objects
            bbox = image_annotation["bbox"]
            if compute_bbox_area(bbox) < class_thresholds[object_class][0]:
                continue

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )
            if distance_to_object <= self.min_interaction_distance:
                continue

            search_object_ids.append(object_type)
            search_object_bboxes.append(image_annotation["bbox"])
            search_object_attributes.append(
                SimBotObjectAttributes(
                    readable_name=readable_name,
                    color=self._object_color_map.get(readable_name, None),
                )
            )

        instructions = []
        if search_object_ids:
            instruction = AugmentationInstruction(
                action_type="Search",
                object_id=search_object_ids,
                attributes=search_object_attributes,
                bbox=search_object_bboxes,
                image_name=image_name,
                annotation_id=0,
                room_name=room_name,
                augmentation_metadata={"positive": True},
            )
            instructions.append(instruction)

        negative_search_object_ids = []
        negative_search_object_attributes = []
        # If there is a searchable object that was not present, this is a negative example
        for search_object, is_present in objects_in_image.items():
            if is_present:
                continue
            negative_search_object_ids.append(search_object)
            negative_search_object_attributes.append(
                SimBotObjectAttributes(
                    readable_name=search_object,
                    color=self._object_color_map.get(search_object, None),
                )
            )

        if negative_search_object_ids:
            instruction = AugmentationInstruction(
                action_type="Search",
                object_id=negative_search_object_ids,
                attributes=negative_search_object_attributes,
                bbox=None,
                image_name=image_name,
                annotation_id=0,
                room_name=room_name,
                augmentation_metadata={"positive": False},
            )
            instructions.append(instruction)
        return instructions

    def post_process_metadata(self, search_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the search actions.

        This basically downsamples the negative examples in the dataset using a fixed maximum
        number of negative examples per room.
        """
        final_metadata: dict[str, Any] = {}
        negative_metadata_grouped_by_room: dict[str, Any] = {}

        for key, annotation in search_metadata.items():
            if annotation["positive"]:
                final_metadata[key] = annotation
            else:
                room = annotation["room_name"]

                negative_metadata = negative_metadata_grouped_by_room.get(room, [])
                negative_metadata.append({key: annotation})
                negative_metadata_grouped_by_room[room] = negative_metadata

        for _, room_metadata in negative_metadata_grouped_by_room.items():
            random.shuffle(room_metadata)
            for room_annotation in room_metadata[: self.max_negative_examples_per_room]:
                final_metadata.update(room_annotation)
        return final_metadata
