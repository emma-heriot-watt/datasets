# flake8: noqa WPS226
import json
import random
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from sklearn.model_selection import StratifiedShuffleSplit

from emma_datasets.common import Settings


settings = Settings()

from pydantic import BaseModel, validator

from emma_datasets.common.logger import get_logger


logger = get_logger(__name__)


class AnnotationModel(BaseModel):
    """A simple model to validate the basic components of an annotation instruction."""

    actions: list[dict[str, Any]]
    instruction: dict[str, Any]
    vision_augmentation: bool

    @validator("instruction")
    @classmethod
    def validate_instruction(cls, field_value: dict[str, Any]) -> dict[str, Any]:
        """Verify instruction is correct."""
        if "instruction" not in field_value or not field_value["instruction"]:
            raise AssertionError(f"Missing instruction in {field_value}")

        single_action = (
            "actions" not in field_value
            or len(field_value["actions"]) > 1
            or field_value["actions"][0] != 0
        )
        if single_action:
            raise AssertionError("Instructions should exactly one action")
        return field_value

    @validator("actions")
    @classmethod
    def validation_actions(  # noqa: WPS231, WPS238
        cls, field_value: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Verify action is correct."""
        if len(field_value) > 1:
            raise AssertionError("Instructions should exactly one action")

        action_type = field_value[0]["type"].lower()
        action_metadata = field_value[0][action_type]
        if "object" not in action_metadata:
            raise AssertionError(f"There is no object property in {field_value}")

        if not isinstance(action_metadata["object"]["id"], str) and action_type != "search":
            raise AssertionError(f"Incorrect object id in {field_value}")

        if not isinstance(action_metadata["object"]["id"], list) and action_type == "search":
            raise AssertionError(f"Incorrect object id in {field_value}")

        if isinstance(action_metadata["object"]["id"][0], list):
            raise AssertionError(f"Incorrect list of object id in {field_value}")

        wrong_mask = (
            "mask" not in action_metadata["object"] or not action_metadata["object"]["mask"]
        )
        if wrong_mask:
            if action_type == "search":
                raise AssertionError(
                    f"Expecting a list of bbox-like mask with a single element {field_value}"
                )
            raise AssertionError(f"Expecting a bbox-like mask {field_value}")

        if action_type == "search":
            if (
                len(action_metadata["object"]["mask"]) != 1
                or len(action_metadata["object"]["mask"][0]) != 4
            ):
                raise AssertionError(
                    f"Expecting a list of bbox-like mask with a single element {field_value}"
                )
        else:
            if len(action_metadata["object"]["mask"]) != 4:
                raise AssertionError(
                    f"Expecting a list of mask with a single element {field_value}"
                )
        return field_value

    @validator("vision_augmentation")
    @classmethod
    def validation_vision_augmentation(cls, field_value: bool) -> bool:
        """Verify the visual_augmentation is correct."""
        if not field_value:
            raise AssertionError("Instances should have the vision_augmentation property to True")
        return field_value


def merge_all_annotations(root_annotation_path: Path) -> dict[str, Any]:  # noqa: WPS231
    """Merge all annotations within the root annotation path in a single annotation dict."""
    merged_annotations = {}
    for annotation_path in root_annotation_path.iterdir():
        if annotation_path.suffix == ".json":
            with open(annotation_path) as fp:
                annotations = json.load(fp)
            annotation_keys = list(annotations.keys())
            if any([key in merged_annotations for key in annotation_keys]):
                raise AssertionError("Found multiple annotations.")
            annotation = annotations[annotation_keys[0]]

            for annotation_key in annotation_keys:
                print(annotation_path, annotation_key)
                annotation = annotations[annotation_key]
                try:
                    AnnotationModel(
                        instruction=annotation["instruction"],
                        actions=annotation["actions"],
                        vision_augmentation=annotation["vision_augmentation"],
                    )
                except Exception:
                    logger.error(f"Skipping {annotation_key}")
                    annotations.pop(annotation_key, None)
            merged_annotations.update(annotations)
    return merged_annotations


def split_annotations(
    annotations: dict[str, Any],
    split: Literal["random", "object_stratified"],
    split_valid_perc: float = 0.2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the annotations."""
    if split == "random":
        annotation_keys = list(annotations.keys())
        random.shuffle(annotation_keys)

        validation_keys = annotation_keys[: int(split_valid_perc * len(annotation_keys))]
        train_keys = annotation_keys[int(split_valid_perc * len(annotation_keys)) :]

        return {key: annotations[key] for key in train_keys}, {
            key: annotations[key] for key in validation_keys
        }

    annotation_keys = list(annotations.keys())
    object_ids = []
    for _, annotation in annotations.items():
        action_type = annotation["actions"][0]["type"].lower()
        action_metadata = annotation["actions"][0][action_type]
        if action_type == "search":
            object_ids.append(action_metadata["object"]["id"][0])
        else:
            object_ids.append(action_metadata["object"]["id"])

    # Objects that have been annotated only once cannot be split in a stratified way.
    # Append them to training only.
    singular_annotations = {}
    objects_counter = Counter(object_ids)
    for object_id, object_counter in objects_counter.items():
        if object_counter == 1:
            # Store the annotation for the object that appeared only once
            index_to_remove = object_ids.index(object_id)
            annotation_key = annotation_keys[index_to_remove]
            singular_annotations[annotation_key] = annotations[annotation_key]

            # Remove the annotation and the object from the split
            annotation_keys.pop(index_to_remove)
            object_ids.pop(index_to_remove)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=split_valid_perc)
    (train_indices, valid_indices) = next(splitter.split(annotation_keys, object_ids))

    train_annotations = {
        annotation_keys[train_idx]: annotations[annotation_keys[train_idx]]
        for train_idx in train_indices
    }

    train_annotations.update(singular_annotations)

    valid_annotations = {
        annotation_keys[valid_idx]: annotations[annotation_keys[valid_idx]]
        for valid_idx in valid_indices
    }
    return train_annotations, valid_annotations


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--root_annotation_path",
        type=Path,
        help="Path to the root directory containing the annotation json files",
    )

    parser.add_argument(
        "--split_train_valid",
        help="Split the annotations into train and validation",
        choices=["random", "object_stratified"],
    )

    parser.add_argument(
        "--split_valid_perc",
        help="Percentage of validation data",
        type=float,
        default=0.2,  # noqa: WPS432
    )

    parser.add_argument(
        "--train_annotation_path",
        type=Path,
        help="Path to output train annotation json",
        default=settings.paths.simbot.joinpath("train_annotation_instructions.json"),
    )

    parser.add_argument(
        "--valid_annotation_path",
        type=Path,
        help="Path to output valid annotation json. Used only when --split-train-val argument is not None.",
        default=settings.paths.simbot.joinpath("valid_annotation_instructions.json"),
    )

    args = parser.parse_args()

    root_annotation_path = args.root_annotation_path

    merged_annotations = merge_all_annotations(root_annotation_path)
    if args.split_train_valid is not None:
        train_annotations, valid_annotations = split_annotations(
            annotations=merged_annotations,
            split=args.split_train_valid,
            split_valid_perc=args.split_valid_perc,
        )

        with open(args.train_annotation_path, "w") as fp_train:
            json.dump(train_annotations, fp_train, indent=4)
        with open(args.valid_annotation_path, "w") as fp_valid:
            json.dump(valid_annotations, fp_valid, indent=4)
    else:
        with open(args.train_annotation_path, "w") as fp_merged:
            json.dump(merged_annotations, fp_merged, indent=4)
