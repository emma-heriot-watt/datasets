import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from emma_datasets.augmentations.simbot_augmentators.base_augmentator import BaseAugmentation
from emma_datasets.augmentations.simbot_augmentators.clip_image_diversity import CLIProcessor
from emma_datasets.common.settings import Settings
from emma_datasets.constants.simbot.simbot import get_objects_asset_synonyms
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_asset_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.object_features_processing import (
    compute_bbox_area,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import SearchParaphraser
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    AugmentationInstruction,
    SimBotObjectAttributes,
)


settings = Settings()


class SearchAugmentation(BaseAugmentation):
    """Search Augmentations."""

    def __init__(
        self,
        root_vision_path: Path,
        report_path: Path,
        diverse_image_selector: CLIProcessor,
        search_classes: list[str],
        action_type: str = "Search",
        min_interaction_distance: float = 0,
        max_negative_examples_per_room: int = 150,
        max_examples_per_object: int = 4000,
    ) -> None:
        super().__init__(root_vision_path, report_path, diverse_image_selector)
        self.min_interaction_distance = min_interaction_distance

        # Force search special monitors
        self.search_objects = search_classes
        self.max_negative_examples_per_room = max_negative_examples_per_room
        self.max_examples_per_object = max_examples_per_object
        self.action_type = action_type

        self._paraphraser = SearchParaphraser(get_objects_asset_synonyms())

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
            if object_type == "Unassigned" or not object_type:
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
                    color=self._get_color(readable_name),
                    distance=distance_to_object,
                )
            )

        instructions = []
        if search_object_ids:
            instruction = AugmentationInstruction(
                action_type=self.action_type,
                object_id=search_object_ids,
                attributes=search_object_attributes,
                bbox=search_object_bboxes,
                image_name=image_name,
                annotation_id=0,
                room_name=room_name,
                augmentation_metadata={"positive": True},
            )
            instructions.append(instruction)

        negative_instruction = self._get_negative_instance(
            objects_in_image=objects_in_image, room_name=room_name, image_name=image_name
        )

        if negative_instruction is not None:
            instructions.append(negative_instruction)
        return instructions

    @classmethod
    def from_yaml_config(  # type: ignore[override]
        cls,
        root_vision_path: Path,
        report_path: Path,
        diverse_image_selector: CLIProcessor,
        search_classes: list[str],
        action_type: str = "Search",
        min_interaction_distance: float = 0,
        max_negative_examples_per_room: int = 150,
        max_examples_per_object: int = 4000,
    ) -> BaseAugmentation:
        """Instantiate the class."""
        return cls(
            root_vision_path=root_vision_path,
            report_path=report_path,
            diverse_image_selector=diverse_image_selector,
            search_classes=search_classes,
            action_type=action_type,
            min_interaction_distance=min_interaction_distance,
            max_negative_examples_per_room=max_negative_examples_per_room,
            max_examples_per_object=max_examples_per_object,
        )

    def post_process_metadata(  # noqa: WPS231
        self, search_metadata: dict[str, Any], class_thresholds: dict[str, list[int]]
    ) -> dict[str, Any]:
        """Post process the metadata for the search actions.

        This basically downsamples the negative examples in the dataset using a fixed maximum
        number of negative examples per room.
        """
        final_metadata: dict[str, Any] = {}
        action_metadata_grouped_per_object_class: dict[str, Any] = {}

        idx = 0
        for key, annotation in search_metadata.items():
            if not annotation["positive"]:
                continue
            action = annotation["actions"][0][self.action_type.lower()]

            object_ids = action["object"]["id"]
            for object_id, object_attributes in zip(object_ids, action["object"]["attributes"]):
                object_asset = get_object_asset_from_object_id(object_id, self._assets_to_labels)
                object_class = self._special_object_type_map.get(
                    object_asset, self._assets_to_labels[object_asset]
                )

                instructions = action_metadata_grouped_per_object_class.get(object_class, [])
                temp_annotation = deepcopy(annotation)
                temp_annotation["actions"][0][self.action_type.lower()]["selected_object"] = {
                    "id": object_id,
                    "attributes": object_attributes,
                }
                temp_annotation["instruction"]["instruction"] = self._paraphraser(
                    object_id=object_id,
                    attributes=SimBotObjectAttributes.parse_obj(object_attributes),
                )
                instructions.append({f"{key}_{idx}": temp_annotation})
                action_metadata_grouped_per_object_class[object_class] = instructions
                idx += 1

        for object_metadata in action_metadata_grouped_per_object_class.values():
            images = [
                instance["actions"][0]["colorImages"][0]
                for metadata in object_metadata
                for instance in metadata.values()
            ]
            _, selected_indices = self.diverse_image_selector(
                images, centroids=self.max_examples_per_object
            )

            for select_index in selected_indices:
                final_metadata.update(object_metadata[select_index])

        self._make_report(final_metadata, class_thresholds)

        negative_metadata_grouped_by_room = self._negative_room_metadata(search_metadata)

        for _, room_metadata in negative_metadata_grouped_by_room.items():
            random.shuffle(room_metadata)
            for room_annotation in room_metadata[: self.max_negative_examples_per_room]:
                final_metadata.update(room_annotation)

        return final_metadata

    def _get_negative_instance(
        self, objects_in_image: dict[str, Any], image_name: str, room_name: str
    ) -> Optional[AugmentationInstruction]:
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
                        color=self._get_color(search_object),
                    )
                )

        if negative_search_object_ids:
            return AugmentationInstruction(
                action_type="Search",
                object_id=negative_search_object_ids,
                attributes=negative_search_object_attributes,
                bbox=None,
                image_name=image_name,
                annotation_id=0,
                room_name=room_name,
                augmentation_metadata={"positive": False},
            )
        return None

    def _negative_room_metadata(self, search_metadata: dict[str, Any]) -> dict[str, Any]:
        idx = 0
        negative_metadata_grouped_by_room: dict[str, Any] = {}
        for key, annotation in search_metadata.items():
            if annotation["positive"]:
                continue

            room = annotation["room_name"]
            action = annotation["actions"][0][self.action_type.lower()]

            object_ids = action["object"]["id"]
            for object_id, object_attributes in zip(object_ids, action["object"]["attributes"]):
                instructions = negative_metadata_grouped_by_room.get(room, [])
                new_annotation = deepcopy(annotation)
                new_annotation["actions"][0][self.action_type.lower()]["selected_object"] = {
                    "id": object_id,
                    "attributes": object_attributes,
                }
                new_annotation["instruction"]["instruction"] = self._paraphraser(
                    object_id=object_id,
                    attributes=SimBotObjectAttributes.parse_obj(object_attributes),
                )
                instructions.append({f"{key}_{idx}": new_annotation})
                negative_metadata_grouped_by_room[room] = instructions
                idx += 1

            negative_metadata = negative_metadata_grouped_by_room.get(room, [])
            negative_metadata.append({key: annotation})
            negative_metadata_grouped_by_room[room] = negative_metadata
        return negative_metadata_grouped_by_room
