import json
from pathlib import Path
from typing import Any, Optional

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance, BaseModel
from emma_datasets.datamodels.constants import MediaType


settings = Settings()


class Ego4DNLQueryItem(BaseModel):
    """The dataclass for an Ego4D Natural Language query annotation item."""

    clip_start_sec: Optional[float]
    clip_end_sec: Optional[float]
    video_start_sec: Optional[float]
    video_end_sec: Optional[float]
    video_start_frame: Optional[int]
    video_end_frame: Optional[int]
    template: Optional[str]
    query: Optional[str]
    slot_x: Optional[str]
    verb_x: Optional[str]
    slot_y: Optional[str]
    verb_y: Optional[str]
    raw_tags: Optional[list[Optional[str]]]


class Ego4DNLAnnotation(BaseModel):
    """The dataclass for an Natural Language Queries annotation."""

    language_queries: list[Ego4DNLQueryItem]


class Ego4DMomentLabel(BaseModel):
    """A dataclass for an Ego4D moment annotation."""

    start_time: float
    end_time: float
    label: str
    video_start_time: float
    video_end_time: float
    video_start_frame: int
    video_end_frame: int
    primary: bool


class Ego4DMomentsAnnotation(BaseModel):
    """A dataclass for a list of Ego4D moment annotations."""

    annotator_uid: str
    labels: list[Ego4DMomentLabel]


class Ego4DInstance(BaseInstance):
    """Base class for all the Ego4D instance classes."""

    video_uid: str
    clip_uid: str
    video_start_sec: float
    video_end_sec: float
    video_start_frame: int
    video_end_frame: int
    clip_start_sec: float
    clip_end_sec: float
    clip_start_frame: int
    clip_end_frame: int
    source_clip_uid: str

    @property
    def modality(self) -> MediaType:
        """Get the modality of the instance."""
        return MediaType.video

    @property
    def features_path(self) -> Path:
        """Get the path to the features for this instance."""
        return Settings().paths.ego4d_features.joinpath(f"{self.clip_uid}.pt")


class Ego4DNLQInstance(Ego4DInstance):
    """The dataclass for an Ego4D Natural Language Queries instance."""

    annotations: list[Ego4DNLAnnotation]


class Ego4DMomentsInstance(Ego4DInstance):
    """The dataclass for an Ego4D Moments Queries instance."""

    annotations: Optional[list[Ego4DMomentsAnnotation]]


class Ego4DResponseTrack(BaseModel):
    """Records information about where an object was seen in the video."""

    frame_number: int
    x: float
    y: float
    width: float
    height: float
    rotation: float
    original_width: float
    original_height: float
    video_frame_number: int


class Ego4DVisualCrop(BaseModel):
    """Dataclass that models an object crop in a video frame."""

    frame_number: int
    x: float
    y: float
    width: float
    height: float
    rotation: float
    original_width: float
    original_height: float
    video_frame_number: int


class Ego4DVQQueryData(BaseModel):
    """Dataclass for a visual query annotation.

    All these arguments are considered Optional because we found some instances that do not have
    all these values.
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    query_frame: Optional[int]
    query_video_frame: Optional[int]
    response_track: Optional[list[Ego4DResponseTrack]]
    object_title: Optional[str]
    visual_crop: Optional[Ego4DVisualCrop]


class Ego4DVQAnnotations(BaseModel):
    """Ego4D visual query annotation for a given example."""

    query_sets: dict[str, Ego4DVQQueryData]
    warnings: list[str]


class Ego4DVQInstance(Ego4DInstance):
    """Ego4D list of visual query annotations."""

    annotations: list[Ego4DVQAnnotations]
    annotation_complete: bool


def load_ego4d_annotations(path: Path) -> list[dict[str, Any]]:
    """Loads the raw Ego4D annotations from the raw JSON file.

    More information about the data format can be found: https://ego4d-data.org/docs/data/annotations-schemas/
    """
    with open(path) as in_file:
        data = json.load(in_file)

    clips_data = []

    for video in data["videos"]:
        for clip in video["clips"]:
            new_clip = clip.copy()

            new_clip["video_uid"] = video["video_uid"]
            clips_data.append(new_clip)

    return clips_data
