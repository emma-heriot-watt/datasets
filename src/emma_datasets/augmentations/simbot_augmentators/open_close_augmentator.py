from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from emma_datasets.augmentations.simbot_augmentators.base_augmentator import BaseAugmentation
from emma_datasets.augmentations.simbot_augmentators.clip_image_diversity import CLIProcessor
from emma_datasets.common import Settings
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_asset_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    AugmentationInstruction,
    SimBotObjectAttributes,
)


settings = Settings()


class OpenCloseAugmentation(BaseAugmentation):
    """OpenClose Augmentations."""

    def __init__(
        self,
        root_vision_path: Path,
        report_path: Path,
        diverse_image_selector: CLIProcessor,
        action_type_classes: list[str],
        action_type: str = "Open",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> None:
        super().__init__(root_vision_path, report_path, diverse_image_selector)
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

            if self._should_skip(object_annotation):
                continue

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
    def from_yaml_config(  # type: ignore[override]
        cls,
        root_vision_path: Path,
        report_path: Path,
        diverse_image_selector: CLIProcessor,
        action_type_classes: list[str],
        action_type: str = "Open",
        min_interaction_distance: float = 1.5,
        max_examples_per_class: int = 5000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            root_vision_path=root_vision_path,
            report_path=report_path,
            diverse_image_selector=diverse_image_selector,
            action_type_classes=action_type_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_examples_per_class=max_examples_per_class,
        )

    def post_process_metadata(
        self, action_metadata: dict[str, Any], class_thresholds: dict[str, list[int]]
    ) -> dict[str, Any]:
        """Post process the metadata for the open and close actions."""
        downsampled_metadata = self._downsample_augmentation_metadata(
            action_type_metadata=action_metadata
        )
        self._make_report(downsampled_metadata, class_thresholds)
        return downsampled_metadata

    def _should_skip(self, object_annotation: dict[str, Any]) -> bool:
        if self.action_type == "Open" and object_annotation["currentStates"]["OPENED"]:
            return True

        return self.action_type == "Close" and not object_annotation["currentStates"]["OPENED"]
