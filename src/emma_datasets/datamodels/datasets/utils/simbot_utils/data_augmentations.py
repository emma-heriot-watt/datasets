import random
from collections import Counter, defaultdict
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


class BaseAugmentation:
    """Base class for object augmentations."""

    def __init__(self) -> None:
        self.action_type = "Base"
        self.max_examples_per_class = 1
        self.action_objects: list[str] = []
        self._assets_to_labels = get_arena_definitions()["asset_to_label"]
        self._labels_to_assets = defaultdict(list)
        for asset, object_label in self._assets_to_labels.items():
            self._labels_to_assets[object_label].append(asset)
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
        self._special_object_class_map = {
            object_class: object_type
            for object_type, object_class in self._special_object_type_map.items()
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

    def _downsample_augmentation_metadata(
        self, action_type_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Downsamples the dataset using a fixed maximum number of examples per object."""
        final_metadata: dict[str, Any] = {}
        action_metadata_grouped_per_object_class: dict[str, Any] = {}

        for key, annotation in action_type_metadata.items():
            action = annotation["actions"][0][self.action_type.lower()]
            object_id = action["object"]["id"]

            object_asset = get_object_asset_from_object_id(object_id, self._assets_to_labels)
            object_class = self._special_object_type_map.get(
                object_asset, self._assets_to_labels[object_asset]
            )

            instructions = action_metadata_grouped_per_object_class.get(object_class, [])
            instructions.append({key: annotation})
            action_metadata_grouped_per_object_class[object_class] = instructions

        for _, object_class_metadata in action_metadata_grouped_per_object_class.items():
            random.shuffle(object_class_metadata)
            for object_class_annotation in object_class_metadata[: self.max_examples_per_class]:
                final_metadata.update(object_class_annotation)
        return final_metadata

    def _should_ignore_annotation_for_image(
        self,
        annotation: dict[str, Any],
        robot_position: NDArray[np.float32],
        class_thresholds: dict[str, list[int]],
    ) -> bool:
        """Check basic stuff to verify that an annotation is valid for an augmentation.

        An annotation is valid if 1) the object class is not `Unassigned`, 2) the object is in the
        actionable objects, and 3) the area bounding box of the object is within the object class
        thresholds.
        """
        image_annotation = annotation["image_annotation"]
        object_type = image_annotation["object_type"]
        if object_type == "Unassigned":
            return True

        object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
        object_class = self._assets_to_labels[object_asset]
        readable_name = self._special_object_type_map.get(object_asset, object_class)

        # Ignore objects that are not specified
        if readable_name not in self.action_objects:
            return True

        # Ignore too small objects
        bbox = self._get_bbox(image_annotation)
        if compute_bbox_area(bbox) < class_thresholds[object_class][0]:  # noqa: WPS531
            return True
        return False

    def _merge_instructions(
        self, action_instructions_dict: dict[str, Any], annotation_id: int
    ) -> list[dict[str, Any]]:
        """Merge the instructions into a list.

        Create additional instructions determined by the object attributes.
        """
        action_instructions = []
        for _, object_instructions in action_instructions_dict.items():
            if len(object_instructions) > 1:
                instructions, annotation_id = self._get_instructions_from_attributes(
                    object_instructions, annotation_id
                )
                action_instructions.extend(instructions)
            else:
                action_instructions.extend(object_instructions)
        return action_instructions

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
            compute_bbox_center_coords(instruction.bbox) for instruction in instruction_list
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

    def _get_bbox(self, image_annotation: dict[str, Any]) -> list[int]:
        return image_annotation["bbox"]


class GoToAugmentation(BaseAugmentation):
    """Goto Augmentations."""

    def __init__(
        self,
        goto_classes: list[str],
        action_type: str = "Goto",
        min_interaction_distance: float = 2.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class
        self.action_type = action_type

        # Force goto special monitors
        self.action_objects = goto_classes + list(self._special_object_type_map.values())

    def __call__(
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
            should_ignore_ann = self._should_ignore_annotation_for_image(
                annotation, robot_position, class_thresholds
            )
            if should_ignore_ann:
                continue
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )
            if distance_to_object > self.min_interaction_distance:
                instruction = AugmentationInstruction(
                    action_type=self.action_type,
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name,
                        color=self._object_color_map.get(readable_name, None),
                        distance=distance_to_object,
                    ),
                    bbox=self._get_bbox(image_annotation),
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )
                annotation_id += 1
                navigation_instructions_dict[object_class].append(instruction)

        navigation_instructions = self._merge_instructions(
            navigation_instructions_dict, annotation_id
        )
        return navigation_instructions

    @classmethod
    def from_yaml_config(
        cls,
        goto_classes: list[str],
        action_type: str = "Goto",
        min_interaction_distance: float = 2.5,
        max_examples_per_class: int = 5000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            goto_classes=goto_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_examples_per_class=max_examples_per_class,
        )

    def post_process_metadata(self, goto_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the goto actions."""
        return self._downsample_augmentation_metadata(action_type_metadata=goto_metadata)


class BreakAugmentation(BaseAugmentation):
    """Break Augmentation."""

    def __init__(
        self,
        break_classes: list[str],
        action_type: str = "Break",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class
        self.action_type = action_type

        self.action_objects = break_classes

    def __call__(
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Get new annotations for the selected classes."""
        break_instructions_dict = defaultdict(list)
        annotation_id = 0
        for _, annotation in annotations.items():
            should_ignore_ann = self._should_ignore_annotation_for_image(
                annotation, robot_position, class_thresholds
            )
            if should_ignore_ann:
                continue
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )

            is_breakable = "BREAKABLE" in object_annotation["supportedStates"]
            is_broken = object_annotation["currentStates"].get("Broken", False)
            can_break_object = all(
                [is_breakable, not is_broken, distance_to_object <= self.min_interaction_distance]
            )

            if can_break_object:
                instruction = AugmentationInstruction(
                    action_type=self.action_type,
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name,
                        color=self._object_color_map.get(readable_name, None),
                        distance=distance_to_object,
                    ),
                    bbox=self._get_bbox(image_annotation),
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )
                annotation_id += 1
                break_instructions_dict[object_class].append(instruction)

        break_instructions = self._merge_instructions(break_instructions_dict, annotation_id)
        return break_instructions

    @classmethod
    def from_yaml_config(
        cls,
        break_classes: list[str],
        action_type: str = "Break",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            break_classes=break_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_examples_per_class=max_examples_per_class,
        )

    def post_process_metadata(self, break_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the break actions."""
        return self._downsample_augmentation_metadata(action_type_metadata=break_metadata)


class CleanAugmentation(BaseAugmentation):
    """Clean Augmentation."""

    def __init__(
        self,
        cleanable_object_types: list[str],
        cleaning_classes: list[str],
        action_type: str = "Clean",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 1000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class
        self.action_type = action_type

        # DIRTY interactable uses CLEAN on KitchenCounterSink_01 that is TOGGLED, resulting in DIRTY being set to false
        self.action_objects = cleaning_classes
        self.cleanable_object_types = cleanable_object_types

    def __call__(
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Get new annotations for the selected classes."""
        clean_instructions_dict = defaultdict(list)
        annotation_id = 0
        for _, annotation in annotations.items():
            should_ignore_ann = self._should_ignore_annotation_for_image(
                annotation, robot_position, class_thresholds
            )
            if should_ignore_ann:
                continue
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )
            if distance_to_object <= self.min_interaction_distance:
                cleaning_object_type = random.choice(self.cleanable_object_types)
                cleaning_object_asset = get_object_asset_from_object_id(
                    object_type, self._assets_to_labels
                )
                cleaning_object_class = self._assets_to_labels[object_asset]
                cleaning_readable_name = self._special_object_type_map.get(
                    cleaning_object_asset, cleaning_object_class
                )

                instruction = AugmentationInstruction(
                    action_type=self.action_type,
                    object_id=cleaning_object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=cleaning_readable_name,
                        color=self._object_color_map.get(readable_name, None),
                        distance=distance_to_object,
                    ),
                    bbox=self._get_bbox(image_annotation),
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )

                annotation_id += 1
                clean_instructions_dict[object_class].append(instruction)

        clean_instructions = self._merge_instructions(clean_instructions_dict, annotation_id)
        return clean_instructions

    @classmethod
    def from_yaml_config(
        cls,
        cleanable_object_types: list[str],
        cleaning_classes: list[str],
        action_type: str = "Clean",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 1000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            cleanable_object_types=cleanable_object_types,
            cleaning_classes=cleaning_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_examples_per_class=max_examples_per_class,
        )

    def post_process_metadata(self, clean_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the clean actions."""
        return self._downsample_augmentation_metadata(action_type_metadata=clean_metadata)


class FillPourAugmentation(BaseAugmentation):
    """FillPour Augmentation."""

    def __init__(
        self,
        fillable_object_types: list[str],
        filling_classes: list[str],
        action_type: str = "Fill",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class
        self.action_type = action_type

        # Fillable interactable with FILLED set to false uses FILL on KitchenCounterSink_01 that has TOGGLED set to true, resulting in fillable interactable having FILLED set to true
        self.fillable_object_types = fillable_object_types
        self.filling_classes = filling_classes

    def __call__(  # noqa: WPS231
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Get new annotations for the selected classes."""
        clean_instructions_dict = defaultdict(list)
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

            # Ignore objects that are not specified for the Fill action
            if self.action_type == "Fill" and readable_name not in self.filling_classes:
                continue

            # Ignore objects that are not specified for the Fill action
            if self.action_type == "Pour" and object_type not in self.fillable_object_types:
                continue

            # Ignore too small objects
            bbox = self._get_bbox(image_annotation)
            if compute_bbox_area(bbox) < class_thresholds[object_class][0]:
                continue

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )
            if distance_to_object <= self.min_interaction_distance:
                if self.action_type == "Fill":
                    object_type = random.choice(self.fillable_object_types)
                    object_asset = get_object_asset_from_object_id(
                        object_type, self._assets_to_labels
                    )
                    object_class = self._assets_to_labels[object_asset]
                    readable_name = self._special_object_type_map.get(object_asset, object_class)

                instruction = AugmentationInstruction(
                    action_type=self.action_type,
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name,
                        color=self._object_color_map.get(readable_name, None),
                        distance=distance_to_object,
                    ),
                    bbox=self._get_bbox(image_annotation),
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )

                annotation_id += 1
                clean_instructions_dict[object_class].append(instruction)

        clean_instructions = self._merge_instructions(clean_instructions_dict, annotation_id)
        return clean_instructions

    @classmethod
    def from_yaml_config(
        cls,
        fillable_object_types: list[str],
        filling_classes: list[str],
        action_type: str = "Fill",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            fillable_object_types=fillable_object_types,
            filling_classes=filling_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_examples_per_class=max_examples_per_class,
        )

    def post_process_metadata(self, clean_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the clean actions."""
        return self._downsample_augmentation_metadata(action_type_metadata=clean_metadata)


class ToggleAugmentation(BaseAugmentation):
    """Toggle Augmentations."""

    def __init__(
        self,
        toggle_classes: list[str],
        action_type: str = "Toggle",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class
        self.action_type = action_type

        # Force toggle special monitors
        self.action_objects = toggle_classes + list(self._special_object_type_map.values())

    def __call__(
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
            should_ignore_ann = self._should_ignore_annotation_for_image(
                annotation, robot_position, class_thresholds
            )
            if should_ignore_ann:
                continue
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )

            if distance_to_object <= self.min_interaction_distance:
                instruction = AugmentationInstruction(
                    action_type=self.action_type,
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name,
                        color=self._object_color_map.get(readable_name, None),
                        distance=distance_to_object,
                    ),
                    bbox=self._get_bbox(image_annotation),
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )
                annotation_id += 1
                toggle_instructions_dict[object_class].append(instruction)

        toggle_instructions = self._merge_instructions(toggle_instructions_dict, annotation_id)
        return toggle_instructions

    @classmethod
    def from_yaml_config(
        cls,
        toggle_classes: list[str],
        action_type: str = "Toggle",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            toggle_classes=toggle_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_examples_per_class=max_examples_per_class,
        )

    def post_process_metadata(self, toggle_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the toggle actions."""
        return self._downsample_augmentation_metadata(action_type_metadata=toggle_metadata)


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

    def __call__(
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
            should_ignore_ann = self._should_ignore_annotation_for_image(
                annotation, robot_position, class_thresholds
            )
            if should_ignore_ann:
                continue
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )

            if distance_to_object <= self.min_interaction_distance:
                instruction = AugmentationInstruction(
                    action_type=self.action_type,
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name,
                        color=self._object_color_map.get(object_class, None),
                        distance=distance_to_object,
                    ),
                    bbox=self._get_bbox(image_annotation),
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )
                annotation_id += 1
                open_instructions_dict[object_class].append(instruction)

        open_instructions = self._merge_instructions(open_instructions_dict, annotation_id)
        return open_instructions

    @classmethod
    def from_yaml_config(
        cls,
        action_type_classes: list[str],
        action_type: str = "Open",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            action_type_classes=action_type_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_examples_per_class=max_examples_per_class,
        )

    def post_process_metadata(self, action_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the open and close actions."""
        return self._downsample_augmentation_metadata(action_type_metadata=action_metadata)


class SearchAugmentation(BaseAugmentation):
    """Search Augmentations."""

    def __init__(
        self,
        search_classes: list[str],
        min_interaction_distance: float = 0,
        max_negative_examples_per_room: int = 150,
        max_examples_per_object: int = 4000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance

        # Force search special monitors
        self.search_objects = search_classes + list(self._special_object_type_map.values())
        self.max_negative_examples_per_room = max_negative_examples_per_room
        self.max_examples_per_object = max_examples_per_object

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
            bbox = self._get_bbox(image_annotation)
            if compute_bbox_area(bbox) < class_thresholds[object_class][0]:
                continue

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )
            if distance_to_object <= self.min_interaction_distance:
                continue

            search_object_ids.append(object_type)
            search_object_bboxes.append(self._get_bbox(image_annotation))
            search_object_attributes.append(
                SimBotObjectAttributes(
                    readable_name=readable_name,
                    color=self._object_color_map.get(readable_name, None),
                    distance=distance_to_object,
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
        if all([not is_present for _, is_present in objects_in_image.items()]):
            for search_object in objects_in_image:
                search_object_id = self._special_object_class_map.get(search_object, None)
                if search_object_id is None:
                    search_object_id = self._labels_to_assets[search_object][0]

                negative_search_object_ids.append(search_object_id)
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

    @classmethod
    def from_yaml_config(
        cls,
        search_classes: list[str],
        min_interaction_distance: float = 0,
        max_negative_examples_per_room: int = 150,
        max_examples_per_object: int = 4000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            search_classes=search_classes,
            min_interaction_distance=min_interaction_distance,
            max_negative_examples_per_room=max_negative_examples_per_room,
            max_examples_per_object=max_examples_per_object,
        )

    def post_process_metadata(  # noqa: WPS231
        self, search_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Post process the metadata for the search actions.

        This basically downsamples the negative examples in the dataset using a fixed maximum
        number of negative examples per room.
        """
        # Step1: Compute the frequencies of each object
        counter_dict = Counter()  # type: ignore[var-annotated]
        readable_name_to_metadata = defaultdict(list)
        for key, annotation in search_metadata.items():
            if not annotation["positive"]:
                continue
            search_object_metadata = annotation["actions"][0]["search"]["object"]
            readable_names = {
                attribute["readable_name"] for attribute in search_object_metadata["attributes"]
            }

            counter_dict.update(Counter(readable_names))
            for readable_name in readable_names:
                readable_name_to_metadata[readable_name].append(key)

        object_frequencies = dict(sorted(counter_dict.items(), key=lambda freq: freq[1]))

        # Step2: Add up to max_examples_per_object per object. Note that in the end the frequency of an object may be beyond max_examples_per_object. This is will happen if an object Y has less than max_examples_per_object but the object X is in the same image and has already appeared max_examples_per_objec times.
        final_metadata: dict[str, Any] = {}
        for readable_name in object_frequencies:  # noqa: WPS440
            keys = readable_name_to_metadata[readable_name]

            keys_already_in_final = set(final_metadata.keys()).intersection(set(keys))
            keys_not_in_final = set(keys).difference(keys_already_in_final)

            # We already have enough images for that object, skip these keys
            if len(keys_already_in_final) > self.max_examples_per_object:
                continue

            remaining_count = self.max_examples_per_object - len(keys_already_in_final)
            for key in keys_not_in_final:  # noqa: WPS440
                if remaining_count == 0:
                    break
                final_metadata[key] = search_metadata[key]
                remaining_count -= 1

        # Step 3: Add random negative images
        negative_metadata_grouped_by_room: dict[str, Any] = {}
        for key, annotation in search_metadata.items():  # noqa: WPS440
            if annotation["positive"]:
                continue

            room = annotation["room_name"]

            negative_metadata = negative_metadata_grouped_by_room.get(room, [])
            negative_metadata.append({key: annotation})
            negative_metadata_grouped_by_room[room] = negative_metadata

        for _, room_metadata in negative_metadata_grouped_by_room.items():
            random.shuffle(room_metadata)
            for room_annotation in room_metadata[: self.max_negative_examples_per_room]:
                final_metadata.update(room_annotation)
        return final_metadata


class PickupAugmentation(BaseAugmentation):
    """Pickup Augmentations."""

    def __init__(
        self,
        pickup_classes: list[str],
        action_type: str = "Pickup",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__()
        self.min_interaction_distance = min_interaction_distance
        self.max_examples_per_class = max_examples_per_class
        self.action_type = action_type

        # Force pickup special monitors
        self.action_objects = pickup_classes

    def __call__(
        self,
        annotations: dict[str, Any],
        robot_position: NDArray[np.float32],
        image_name: str,
        class_thresholds: dict[str, list[int]],
        room_name: str,
    ) -> list[AugmentationInstruction]:
        """Get new annotations for the selected classes."""
        pickup_instructions_dict = defaultdict(list)
        annotation_id = 0
        for _, annotation in annotations.items():
            should_ignore_ann = self._should_ignore_annotation_for_image(
                annotation, robot_position, class_thresholds
            )
            if should_ignore_ann:
                continue
            image_annotation = annotation["image_annotation"]
            object_annotation = annotation["object_annotation"]
            object_type = image_annotation["object_type"]

            object_asset = get_object_asset_from_object_id(object_type, self._assets_to_labels)
            object_class = self._assets_to_labels[object_asset]
            readable_name = self._special_object_type_map.get(object_asset, object_class)

            distance_to_object = self._compute_distance_to_object(
                object_annotation, robot_position
            )

            if distance_to_object <= self.min_interaction_distance:
                instruction = AugmentationInstruction(
                    action_type=self.action_type,
                    object_id=object_type,
                    attributes=SimBotObjectAttributes(
                        readable_name=readable_name,
                        color=self._object_color_map.get(readable_name, None),
                        distance=distance_to_object,
                    ),
                    bbox=self._get_bbox(image_annotation),
                    image_name=image_name,
                    annotation_id=annotation_id,
                    room_name=room_name,
                )
                annotation_id += 1
                pickup_instructions_dict[object_class].append(instruction)

        pickup_instructions = self._merge_instructions(pickup_instructions_dict, annotation_id)
        return pickup_instructions

    @classmethod
    def from_yaml_config(
        cls,
        pickup_classes: list[str],
        action_type: str = "Pickup",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            pickup_classes=pickup_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_examples_per_class=max_examples_per_class,
        )

    def post_process_metadata(self, pickup_metadata: dict[str, Any]) -> dict[str, Any]:
        """Post process the metadata for the pickup actions."""
        return self._downsample_augmentation_metadata(action_type_metadata=pickup_metadata)
