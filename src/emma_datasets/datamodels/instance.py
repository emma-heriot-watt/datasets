from typing import Optional

import numpy
from numpy.typing import NDArray

from emma_datasets.datamodels.base_model import BaseModel
from emma_datasets.datamodels.constants import DatasetModalityMap, DatasetName, MediaType
from emma_datasets.datamodels.dataset_metadata import DatasetMetadata
from emma_datasets.datamodels.region import Region
from emma_datasets.datamodels.scene_graph import SceneGraph
from emma_datasets.datamodels.text import Caption, QuestionAnswerPair
from emma_datasets.datamodels.trajectory import ActionTrajectory


Pixels = NDArray[numpy.float32]

DatasetDict = dict[DatasetName, DatasetMetadata]


class Instance(BaseModel):
    """Instance within the dataset."""

    # id: str
    dataset: DatasetDict
    caption: Optional[Caption]
    qa: Optional[QuestionAnswerPair]
    regions: Optional[list[Region]]
    scene_graph: Optional[SceneGraph]
    trajectory: Optional[ActionTrajectory]

    @property
    def modality(self) -> MediaType:
        """Returns the modality of the instance."""
        instance_modalities = {
            DatasetModalityMap[dataset_name] for dataset_name in self.dataset.keys()
        }

        if len(instance_modalities) > 1:
            return max(instance_modalities)

        return next(iter(instance_modalities))
