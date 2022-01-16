import ast
from datetime import datetime
from typing import Any, Optional, Union

from pydantic import validator

from src.datamodels.base_model import BaseModel
from src.datamodels.constants import DatasetSplit


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
