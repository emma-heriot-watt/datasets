import json
import random
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from sklearn.model_selection import StratifiedShuffleSplit

from emma_datasets.common import Settings


settings = Settings()


def merge_all_annotations(root_annotation_path: Path) -> dict[str, Any]:
    """Merge all annotations within the root annotation path in a single annotation dict."""
    merged_annotations = {}
    for annotation_path in root_annotation_path.iterdir():
        if annotation_path.suffix == ".json":
            with open(annotation_path) as fp:
                annotations = json.load(fp)
            annotation_keys = list(annotations.keys())
            if any([key in merged_annotations for key in annotation_keys]):
                raise AssertionError("Found multiple annotations.")
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
