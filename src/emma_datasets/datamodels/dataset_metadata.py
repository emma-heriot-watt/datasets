from pathlib import Path
from typing import Optional, Union

from pydantic import HttpUrl

from emma_datasets.datamodels.base_model import BaseModel
from emma_datasets.datamodels.constants import DatasetName, DatasetSplit, MediaType


class SourceMedia(BaseModel, frozen=True):
    """Source media from dataset."""

    url: Optional[HttpUrl]
    media_type: MediaType
    path: Optional[Path]
    width: int
    height: int


class DatasetMetadata(BaseModel, frozen=True):
    """Source dataset metadata per instance."""

    id: str
    name: DatasetName
    split: Optional[DatasetSplit] = None
    media: Union[SourceMedia, list[SourceMedia]]
    features_path: Union[Path, list[Path]]

    # From splitters
    scene_graph_path: Optional[Path] = None
    regions_path: Optional[Path] = None
    caption_path: Optional[Path] = None
    qa_pairs_path: Optional[Path] = None
    action_trajectory_path: Optional[Path] = None
    task_description_path: Optional[Path] = None

    @property
    def paths(self) -> Union[Path, list[Path], None]:
        """Get paths to the source media."""
        if isinstance(self.media, list):
            all_paths = [media.path for media in self.media if media.path is not None]
            return all_paths if all_paths else None

        return self.media.path
