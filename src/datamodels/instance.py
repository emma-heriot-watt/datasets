from typing import Iterable, Optional

import numpy
from numpy.typing import NDArray

from src.datamodels.base_model import BaseModel
from src.datamodels.constants import DatasetName, MediaType
from src.datamodels.dataset_metadata import DatasetMetadata
from src.datamodels.region import Region
from src.datamodels.scene_graph import SceneGraph
from src.datamodels.text import Caption, QuestionAnswerPair


Pixels = NDArray[numpy.floating]


class Scene(BaseModel):
    """Common model used by all modalities."""

    media_type: MediaType
    dataset: dict[DatasetName, DatasetMetadata]


class Instance(Scene):
    """Instance within the dataset."""

    # id: str
    caption: Optional[Caption]
    qa: Optional[QuestionAnswerPair]
    regions: Optional[Iterable[Region]]
    scene_graph: Optional[SceneGraph]
