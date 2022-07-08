import pickle  # noqa: S403
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, HttpUrl, PrivateAttr

from emma_datasets.common import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetSplit, MediaType
from emma_datasets.io import read_json


settings = Settings()


class RefCocoRegion(BaseModel):
    """RefCOCO region."""

    annotation_id: str
    image_id: str
    x: float
    y: float
    w: float
    h: float


class RefCocoImageMetadata(BaseModel, frozen=True):
    """Image metadata for RefCOCO scene."""

    image_id: str
    width: int
    height: int
    url: HttpUrl


class RefCocoExpression(BaseModel):
    """RefCOCO referring expression."""

    sentence: str
    sentence_id: str
    annotation_id: str


def read_refcoco_referring_expressions(
    referring_expressions_path: Path,
) -> dict[DatasetSplit, list[RefCocoExpression]]:
    """Read the RefCOCO referring expressions and group them per split."""
    with open(referring_expressions_path, "rb") as in_file:
        annotations = pickle.load(in_file)  # noqa: S301

    referring_expressions = defaultdict(list)

    for instance in annotations:
        # Get the split of the instance, because all referring expressions are stored in a single file
        if instance["split"] == "val":
            instance["split"] = "valid"
        split = DatasetSplit[instance["split"]]
        # Each instance is associated with multiple referring expressions
        for sentence in instance["sentences"]:
            referring_expressions[split].append(
                RefCocoExpression(
                    sentence=sentence["raw"],
                    sentence_id=str(sentence["sent_id"]),
                    annotation_id=str(instance["ann_id"]),
                )
            )
    return referring_expressions


def read_refcoco_image_metadata(annotation_path: Path) -> dict[str, RefCocoImageMetadata]:
    """Read the metadata for the RefCOCO images.

    Return metadata as a dictionary with image ids as the keys.
    """
    data = read_json(annotation_path)["images"]

    image_metadata = {}
    for image in data:
        image_metadata[str(image["id"])] = RefCocoImageMetadata(
            image_id=str(image["id"]),
            width=image["width"],
            height=image["height"],
            url=image["coco_url"],
        )
    return image_metadata


def read_refcoco_region_annotations(annotation_path: Path) -> dict[str, RefCocoRegion]:
    """Read the annotations for the regions associated with referring expressions.

    The bbox cooridinates are [x,y,w,h] where xy are the cooridinates of the bottom left corner.
    Return metadata as a dictionary with annotation ids as the keys.
    """
    data = read_json(annotation_path)["annotations"]
    regions = {}
    for datum in data:
        regions[str(datum["id"])] = RefCocoRegion(
            annotation_id=str(datum["id"]),
            image_id=str(datum["image_id"]),
            x=datum["bbox"][0],
            y=datum["bbox"][1],
            w=datum["bbox"][2],
            h=datum["bbox"][3],
        )

    return regions


def get_refcoco_paths(refcoco_base_dir: Path) -> tuple[Path, Path]:
    """Get the paths to referring expressions and image annotations."""
    referring_expressions_path = refcoco_base_dir.joinpath("refs(umd).p")
    image_annotations_path = refcoco_base_dir.joinpath("instances.json")
    return referring_expressions_path, image_annotations_path


def merge_refcoco_annotations(
    referring_expressions: list[RefCocoExpression],
    regions_metadata: dict[str, RefCocoRegion],
    image_metadata: dict[str, RefCocoImageMetadata],
) -> list[dict[str, Any]]:
    """Merge the referring expressions, region and image annotations."""
    annotations = []
    for referring_expression in referring_expressions:
        annotation_id = referring_expression.annotation_id
        region = regions_metadata.get(annotation_id)
        if not region:
            continue

        image_id = regions_metadata[annotation_id].image_id
        image = image_metadata.get(image_id)
        if not image_id:
            continue
        instance = {
            "referring_expression": referring_expression,
            "region": region,
            "image_metadata": image,
        }
        annotations.append(instance)

    return annotations


def load_refcoco_annotations(refcoco_base_dir: Path) -> dict[DatasetSplit, Any]:
    """Load the RefCOCOg (UMD) annotations."""
    referring_expressions_path, image_annotations_path = get_refcoco_paths(refcoco_base_dir)
    referring_expressions = read_refcoco_referring_expressions(referring_expressions_path)
    regions_metadata = read_refcoco_region_annotations(image_annotations_path)
    image_metadata = read_refcoco_image_metadata(image_annotations_path)

    # Merge the annotations per split
    annotations = {}
    for split, split_referring_expressions in referring_expressions.items():
        annotations[split] = merge_refcoco_annotations(
            split_referring_expressions, regions_metadata, image_metadata
        )

    return annotations


class RefCocoInstance(BaseInstance):
    """RefCOCO instance."""

    image_metadata: RefCocoImageMetadata
    region: RefCocoRegion
    referring_expression: RefCocoExpression
    _features_path: Path = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        self._features_path = settings.paths.coco_features.joinpath(  # noqa: WPS601
            f"{self.image_metadata.image_id.zfill(12)}.pt"  # noqa: WPS432
        )

    @property
    def modality(self) -> MediaType:
        """Get the modality of the instance."""
        return MediaType.image

    @property
    def features_path(self) -> Path:
        """Get the path to the features for this instance."""
        return self._features_path
