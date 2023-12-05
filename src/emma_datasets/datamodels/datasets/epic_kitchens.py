import ast
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import validator

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance, BaseModel
from emma_datasets.datamodels.constants import DatasetSplit, MediaType


settings = Settings()


def fix_timestamp_fields(timestamp: Union[str, datetime]) -> datetime:
    """Convert strings to time fields."""
    if isinstance(timestamp, str):
        return datetime.strptime(timestamp, "%H:%M:%S.%f")
    return timestamp


def fix_lists(list_object: Union[str, list[Any]]) -> list[Any]:
    """Convert list strings to python lists."""
    if isinstance(list_object, str):
        return ast.literal_eval(list_object)
    return list_object


class EpicKitchensNarrationMetadata(BaseModel):
    """Metadata for a single subgoal from EpicKitchens."""

    narration_id: str
    participant_id: str
    video_id: str
    # narration_timestamp: datetime
    start_timestamp: datetime
    stop_timestamp: datetime
    start_frame: int
    stop_frame: int
    narration: str
    verb: str
    verb_class: int
    noun: str
    noun_class: int
    all_nouns: list[str]
    all_noun_classes: list[int]
    dataset_split: Optional[DatasetSplit]

    # _fix_narrration_timestamp = validator("narration_timestamp", pre=True, allow_reuse=True)(
    #     fix_timestamp_fields
    # )
    _fix_start_timestamp = validator("start_timestamp", pre=True, allow_reuse=True)(
        fix_timestamp_fields
    )
    _fix_stop_timestamp = validator("stop_timestamp", pre=True, allow_reuse=True)(
        fix_timestamp_fields
    )

    _fix_all_nouns = validator("all_nouns", pre=True, allow_reuse=True)(fix_lists)
    _fix_all_noun_classes = validator("all_noun_classes", pre=True, allow_reuse=True)(fix_lists)


class EpicKitchensInstance(BaseInstance):
    """The dataclass for an EpicKitchen instance."""

    narration_id: str
    participant_id: str
    video_id: str
    narration_timestamp: str
    start_timestamp: str
    stop_timestamp: str
    start_frame: Optional[str]
    stop_frame: Optional[str]
    narration: Optional[str]
    verb: Optional[str]
    verb_class: Optional[str]
    noun: Optional[str]
    noun_class: Optional[str]
    all_nouns: Optional[list[str]]
    all_noun_classes: Optional[list[int]]

    _fix_all_nouns = validator("all_nouns", pre=True, allow_reuse=True)(fix_lists)
    _fix_all_noun_classes = validator("all_noun_classes", pre=True, allow_reuse=True)(fix_lists)

    @property
    def modality(self) -> MediaType:
        """Get the modality of the instance."""
        return MediaType.video

    @property
    def features_path(self) -> Path:
        """Get the path to the features for this instance."""
        return Settings().paths.epic_kitchens_features.joinpath(f"{self.narration_id}.pt")
