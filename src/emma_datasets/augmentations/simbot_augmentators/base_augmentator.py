import abc
import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from emma_datasets.augmentations.simbot_augmentators.clip_image_diversity import CLIProcessor
from emma_datasets.common import Settings, use_rich_for_logging
from emma_datasets.constants.simbot.simbot import get_arena_definitions
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
use_rich_for_logging()
logger = logging.getLogger(__name__)


class BaseAugmentation(abc.ABC):
    """Base class for object augmentations."""

    def __init__(
        self,
        root_vision_path: Path,
        report_path: Path,
        diverse_image_selector: CLIProcessor,
    ) -> None:
        self.root_vision_path = root_vision_path
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
            "Embiggenator Monitor": ["pink", "purple"],
            "Freeze Ray Monitor": ["blue"],
            "Freeze Ray Shelf": ["blue"],
            "Gravity Monitor": ["green"],
            "Portal Generator Monitor": ["black"],
            "Laser Monitor": ["red"],
            "Laser Shelf": ["red"],
        }

        self._special_object_type_map = {
            "AP_Prop_Shelf_Wall_04": "Freeze Ray Shelf",
            "AP_Prop_Shelf_Wall_FreezeRay": "Freeze Ray Shelf",
            "AP_Prop_Shelf_Wall_Laser": "Laser Shelf",
            "V_Monitor_Embiggenator": "Embiggenator Monitor",
            "V_Monitor_FreezeRay": "Freeze Ray Monitor",
            "V_Monitor_Gravity": "Gravity Monitor",
            "V_Monitor_Laser": "Laser Monitor",
            "V_Monitor_Portal": "Portal Generator Monitor",
            "Bookshelf_Wooden_01": "Bookshelf",
            "TAMPrototypeHead_01": "Emotion Tester",
            "PackingBox": "Packing Box",
            "CandyJar_01": "Candy Jar",
        }
        self._special_object_class_map = {
            object_class: object_type
            for object_type, object_class in self._special_object_type_map.items()
        }

        self._object_to_special_object_map = dict(self._special_object_type_map.items())

        self.report_path = report_path
        with open(self.report_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["action", "object", "total", "min area", "mean area", "max area"])

        self.diverse_image_selector = diverse_image_selector

    @abc.abstractmethod
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

    @classmethod
    def from_yaml_config(
        cls,
        root_vision_path: Path,
        report_path: Path,
        diverse_image_selector: CLIProcessor,
        *args: Any,
    ) -> "BaseAugmentation":
        """Initiate from config."""
        raise NotImplementedError("Do not call BaseAugmentation class")

    def post_process_metadata(
        self,
        action_type_metadata: dict[str, Any],
        class_thresholds: dict[str, list[int]],
    ) -> dict[str, Any]:
        """Post-process any annotation."""
        return action_type_metadata

    def _downsample_augmentation_metadata(
        self,
        action_type_metadata: dict[str, Any],
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
            images = [
                instance["actions"][0]["colorImages"][0]
                for metadata in object_class_metadata
                for instance in metadata.values()
            ]
            _, selected_indices = self.diverse_image_selector(
                images, centroids=self.max_examples_per_class
            )
            for idx in selected_indices:
                final_metadata.update(object_class_metadata[idx])

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
        if object_type == "Unassigned" or not object_type:
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
        return [dict(instruction) for instruction in action_instructions]

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

    def _get_bbox(self, image_annotation: dict[str, Any]) -> list[int]:
        return image_annotation["bbox"]

    def _get_color(self, readable_name: str) -> Optional[str]:
        colors = self._object_color_map.get(readable_name, None)
        if colors is None:
            return None
        return random.choice(colors)

    def _make_report(
        self, action_metadata: dict[str, Any], class_thresholds: dict[str, list[int]]
    ) -> None:
        report_dict = defaultdict(list)
        for _, annotation in action_metadata.items():
            action_object_metadata = annotation["actions"][-1][self.action_type.lower()]
            if self.action_type == "Search":
                object_id = action_object_metadata["selected_object"]["id"]
                mask_idx = action_object_metadata["object"]["id"].index(object_id)
                bbox = action_object_metadata["object"]["mask"][mask_idx]
                readable_name = action_object_metadata["selected_object"]["attributes"][
                    "readable_name"
                ]
            else:
                bbox = action_object_metadata["object"]["mask"]
                readable_name = action_object_metadata["object"]["attributes"]["readable_name"]
            area = compute_bbox_area(bbox)
            report_dict[readable_name].append(area)

        rows = []
        sorted_readable_names = sorted(report_dict.keys())
        for readable_name in sorted_readable_names:  # noqa: WPS440
            areas = report_dict[readable_name]
            object_asset = self._special_object_class_map.get(readable_name, readable_name)
            object_class = self._assets_to_labels.get(object_asset, object_asset)
            min_threshold = class_thresholds[object_class][0]
            action_msg = f"{self.action_type} {readable_name}:"
            area_msg = f"Total {len(areas)} Min {np.min(areas)} Mean {np.mean(areas)} Max {np.max(areas)} Min Thresh {min_threshold}"  # noqa: WPS221
            logger.info(f"{action_msg} {area_msg}")

            rows.append(
                [
                    self.action_type,
                    readable_name,
                    len(areas),
                    np.min(areas),
                    np.mean(areas),
                    np.max(areas),
                ]
            )

        with open(self.report_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
