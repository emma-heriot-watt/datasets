from typing import Optional, Union

from pydantic import HttpUrl

from emma_datasets.datamodels.base_model import BaseModel
from emma_datasets.datamodels.constants import DatasetName, DatasetSplit, MediaType


class SourceMedia(BaseModel, frozen=True):
    """Source media from dataset."""

    url: Optional[HttpUrl]
    media_type: MediaType
    path: Optional[str]


class DatasetMetadata(BaseModel, frozen=True):
    """Source dataset metadata per instance."""

    id: str
    name: DatasetName
    split: Optional[DatasetSplit] = None
    media: Union[SourceMedia, list[SourceMedia]]
    scene_graph_path: Optional[str] = None
    regions_path: Optional[str] = None
    caption_path: Optional[str] = None
    qa_pairs_path: Optional[str] = None
    action_trajectory_path: Optional[str] = None
