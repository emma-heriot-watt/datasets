from typing import Optional

from pydantic import BaseModel

from emma_datasets.datamodels.constants import DatasetSplit


class GqaImageMetadata(BaseModel, frozen=True):
    """Image metadata from GQA.

    These are extracted from the scene graph JSON files.

    Note:
        - `dataset_split` is not given within the raw metadata for COCO, but should be provided to
            make life easier later on.
        - `id` is a string and not an integer to remain compatible with other datasets.
    """

    id: str
    height: int
    width: int
    file_name: str
    dataset_split: Optional[DatasetSplit]


class GqaRelation(BaseModel):
    """GQA basic relation."""

    name: str
    object: str


class GqaObject(BaseModel):
    """An object in a GQA scene graph."""

    name: str
    attributes: Optional[list[str]]
    x: int
    y: int
    w: int
    h: int
    relations: Optional[list[GqaRelation]]


class GqaSceneGraph(BaseModel):
    """The structure of a GQA scene graph."""

    image_id: str
    width: int
    height: int
    location: Optional[str]
    weather: Optional[str]
    objects: dict[str, GqaObject]  # noqa: WPS110
