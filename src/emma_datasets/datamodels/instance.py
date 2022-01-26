from pathlib import Path
from typing import Optional, Union

from emma_datasets.datamodels.base_model import BaseModel
from emma_datasets.datamodels.constants import DatasetModalityMap, DatasetName, MediaType
from emma_datasets.datamodels.dataset_metadata import DatasetMetadata
from emma_datasets.datamodels.region import Region
from emma_datasets.datamodels.scene_graph import SceneGraph
from emma_datasets.datamodels.text import Caption, QuestionAnswerPair
from emma_datasets.datamodels.trajectory import ActionTrajectory


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

    @property
    def paths(self) -> Union[Path, list[Path], None]:
        """Get source paths for this instance.

        Since an instance can be mapped to more than one dataset, we assume that the source media
        is going to be identical across them. Therefore, it doesn't matter which dataset's image
        file we use since they should be identical.
        """
        return next(iter(self.dataset.values())).paths
