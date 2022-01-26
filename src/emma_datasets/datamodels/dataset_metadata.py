from dataclasses import field
from pathlib import Path
from typing import Optional, Union

from pydantic import HttpUrl
from pydantic.dataclasses import dataclass

from emma_datasets.datamodels.base_model import BaseModel
from emma_datasets.datamodels.constants import Annotation, DatasetName, DatasetSplit, MediaType


class SourceMedia(BaseModel, frozen=True):
    """Source media from dataset."""

    url: Optional[HttpUrl]
    media_type: MediaType
    path: Optional[Path]


@dataclass(frozen=True, unsafe_hash=True, eq=True)
class DatasetMetadata:
    """Source dataset metadata per instance."""

    id: str
    name: DatasetName
    split: Optional[DatasetSplit]
    media: Union[SourceMedia, list[SourceMedia]]
    annotation_paths: dict[Annotation, Path] = field(compare=False)

    @property
    def paths(self) -> Union[Path, list[Path], None]:
        """Get paths to the source media."""
        if isinstance(self.media, list):
            all_paths = [media.path for media in self.media if media.path is not None]
            return all_paths if all_paths else None

        return self.media.path
