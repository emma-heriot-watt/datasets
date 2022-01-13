from typing import Any, Optional, Union

from pydantic import HttpUrl, validator

from src.datamodels.base_model import BaseModel
from src.datamodels.constants import DatasetName, DatasetSplit, MediaType
from src.parsers.helpers import get_image_md5


class SourceMedia(BaseModel):
    """Source media from dataset."""

    url: Optional[HttpUrl]
    media_type: MediaType
    path: str
    md5: str = ""

    @validator("md5")
    @classmethod
    def get_md5_if_not_given(cls, md5: str, values: dict[str, Any]) -> str:  # noqa: WPS110
        """Get MD5 of image if not given."""
        if not md5:
            md5 = get_image_md5(values["path"])
        return md5


class DatasetMetadata(BaseModel):
    """Source dataset metadata per instance."""

    id: str
    name: DatasetName
    split: Optional[DatasetSplit] = None
    media: Union[SourceMedia, list[SourceMedia]]
    scene_graph_path: Optional[str] = None
    regions_path: Optional[str] = None
    caption_path: Optional[str] = None
    qa_pairs_path: Optional[str] = None
