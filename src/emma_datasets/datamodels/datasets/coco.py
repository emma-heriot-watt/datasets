from datetime import datetime
from typing import Optional

from pydantic import BaseModel, HttpUrl

from emma_datasets.datamodels.constants import DatasetSplit


class CocoImageMetadata(BaseModel, frozen=True):
    """Image metadata from COCO.

    Commonly found in the `images` key within the captions JSON files.

    Note:
        - `dataset_split` is not given within the raw metadata for COCO, but should be provided to
            make life easier later on.
        - `id` is a string and not an integer to remain compatible with other datasets.
    """

    id: str
    license: int
    file_name: str
    coco_url: HttpUrl
    flickr_url: HttpUrl
    height: int
    width: int
    date_captured: datetime
    dataset_split: Optional[DatasetSplit]


class CocoCaption(BaseModel):
    """Caption data from COCO.

    Commonly found in the `annotations` key within the captions JSON files.

    Note:
        - `id` is a string and not an integer to remain compatible with other datasets.
    """

    id: str
    image_id: str
    caption: str
