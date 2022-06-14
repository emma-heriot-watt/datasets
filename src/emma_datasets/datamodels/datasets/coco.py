from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, HttpUrl, PrivateAttr

from emma_datasets.common import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetSplit, MediaType


settings = Settings()


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


class CocoInstance(BaseInstance):
    """COCO Instance."""

    image_id: str
    captions_id: list[str]
    captions: list[str]
    _features_path: Path = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        self._features_path = settings.paths.coco_features.joinpath(  # noqa: WPS601
            f"{self.image_id}.pt"
        )

    @property
    def modality(self) -> MediaType:
        """Get the modality of the instance."""
        return MediaType.image

    @property
    def features_path(self) -> Path:
        """Get the path to the features for this instance."""
        return self._features_path
