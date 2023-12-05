import json
from itertools import groupby
from pathlib import Path
from typing import Union

from emma_datasets.datamodels.datasets.coco import CocoInstance


def test_can_load_coco_caption_data(coco_instances_path: Path) -> None:
    assert coco_instances_path.exists()

    with open(coco_instances_path) as in_file:
        annotations = json.load(in_file)["annotations"]

    grouped_annotations: dict[int, dict[str, Union[str, list[str]]]] = {}  # noqa: WPS234
    groups = groupby(annotations, key=lambda x: x["image_id"])
    for image_id, grouped_image_annotations in groups:
        image_annotations = list(grouped_image_annotations)
        grouped_annotations[image_id] = {
            "image_id": str(image_id),
            "captions_id": [str(example["id"]) for example in image_annotations],
            "captions": [example["caption"] for example in image_annotations],
        }

    assert annotations, "The file doesn't contain any instances."

    for _, group_ann in grouped_annotations.items():
        parsed_instance = CocoInstance.parse_obj(group_ann)

        assert parsed_instance.image_id
        assert parsed_instance.captions_id
        assert parsed_instance.captions
