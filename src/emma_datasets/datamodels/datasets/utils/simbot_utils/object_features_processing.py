from pathlib import Path
from typing import Any, Optional

import torch

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_label_from_object_id,
    get_object_readable_name_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.masks import compress_simbot_mask
from emma_datasets.io import read_json


settings = Settings()


class ObjectClassDecoder:
    """Handle the detected objects for a given frame."""

    def __init__(self) -> None:
        arena_definitions = read_json(
            settings.paths.constants.joinpath("simbot/arena_definitions.json")
        )
        self.idx_to_label = {
            idx: label for label, idx in arena_definitions["label_to_idx"].items()
        }
        self._object_assets_to_names = arena_definitions["asset_to_label"]
        self._special_name_cases = arena_definitions["special_asset_to_readable_name"]

    def get_target_object(self, action: dict[str, Any]) -> str:
        """Get the target object id for an action."""
        action_type = action["type"].lower()
        return action[action_type]["object"]["id"]

    def get_target_object_and_name(self, action: dict[str, Any]) -> tuple[str, str, str]:
        """Get the target object id and name for an action."""
        target_object = self.get_target_object(action)
        target_class_label = get_object_label_from_object_id(
            target_object, self._object_assets_to_names
        )
        target_readable_name = get_object_readable_name_from_object_id(
            target_object, self._object_assets_to_names, self._special_name_cases
        )
        return target_object, target_class_label, target_readable_name

    def get_candidate_object_in_frame(
        self,
        mission_id: str,
        action_id: int,
        frame_index: int,
        target_class_label: str,
    ) -> list[int]:
        """Get a list of object indices matching the target object name."""
        features = self.load_features(
            mission_id=mission_id, action_id=action_id, frame_index=frame_index
        )
        if not features:
            return []
        candidate_objects = self._get_candidate_objects_from_features(
            features=features, target_class_label=target_class_label
        )
        if target_class_label == "Shelf":
            candidate_objects.extend(
                self._get_candidate_objects_from_features(
                    features=features, target_class_label="Wall Shelf"
                )
            )
        elif target_class_label in "Cabinet":
            candidate_objects.extend(
                self._get_candidate_objects_from_features(
                    features=features, target_class_label="Counter"
                )
            )
        elif target_class_label == "Box":
            candidate_objects.extend(
                self._get_candidate_objects_from_features(
                    features=features, target_class_label="Cereal Box"
                )
            )
            candidate_objects.extend(
                self._get_candidate_objects_from_features(
                    features=features, target_class_label="Boxes"
                )
            )
        return candidate_objects

    def get_target_object_mask(
        self, mission_id: str, action_id: int, frame_index: int, target_class_label: str
    ) -> Optional[list[list[int]]]:
        """Get the mask of an object that matches the target object name."""
        # Load the features from the Goto action
        features = self.load_features(
            mission_id=mission_id, action_id=action_id, frame_index=frame_index
        )
        if not features:
            return None
        candidate_objects = self._get_candidate_objects_from_features(
            features=features, target_class_label=target_class_label
        )

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

    def load_features(
        self, mission_id: str, action_id: int, frame_index: int
    ) -> Optional[dict[str, Any]]:
        """Get the mask of an object that matches the target object name."""
        # Load the features from the Goto action
        features_path = settings.paths.simbot_features.joinpath(
            f"{mission_id}_action{action_id}.pt"
        )
        if not features_path.exists():
            return None
        return self._load_frame_features(features_path=features_path, frame_index=frame_index)

    def _load_frame_features(self, features_path: Path, frame_index: int) -> dict[str, Any]:
        features = torch.load(features_path)["frames"][frame_index]["features"]
        return features

    def _get_frame_class_indices(self, features: dict[str, Any]) -> list[int]:
        """Get the class indices for the predicted boxes."""
        class_indices = torch.argmax(features["bbox_probas"], dim=1).tolist()
        return class_indices

    def _get_frame_classes(self, features: dict[str, Any]) -> list[str]:
        """Get the class names for the predicted boxes."""
        class_indices = self._get_frame_class_indices(features)
        classes = [self.idx_to_label[class_idx] for class_idx in class_indices]
        return classes

    def _get_candidate_objects_from_features(
        self,
        features: dict[str, Any],
        target_class_label: str,
    ) -> list[int]:
        class_indices = self._get_frame_class_indices(features=features)
        # Get the indices of the objects that match the target_class_label
        candidate_objects = [
            idx
            for idx, class_idx in enumerate(class_indices)
            if self.idx_to_label[class_idx] == target_class_label
        ]
        return candidate_objects


def compute_bbox_center_coords(bbox: list[int]) -> tuple[float, float]:
    """Compute the centre of the bounding box."""
    (x_min, y_min, x_max, y_max) = bbox
    return (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2)


def compute_bbox_area(bbox: list[int]) -> float:
    """Compute the area of the bounding box."""
    return (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
