from typing import Optional

from pydantic import BaseModel, HttpUrl

from emma_datasets.datamodels.constants import DatasetSplit


class VgImageMetadata(BaseModel, frozen=True):
    """Image metadata for Visual Genome scene."""

    image_id: str
    width: int
    height: int
    coco_id: Optional[str]
    flickr_id: Optional[str]
    url: HttpUrl
    dataset_split: Optional[DatasetSplit]


class VgRegion(BaseModel):
    """Visual Genome region."""

    region_id: str
    width: int
    height: int
    image_id: str
    phrase: str
    y: int
    x: int


class VgImageRegions(BaseModel):
    """Regions for Visual Genome Image."""

    id: str
    regions: list[VgRegion]
