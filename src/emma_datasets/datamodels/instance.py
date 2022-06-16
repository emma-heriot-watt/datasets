from pathlib import Path
from typing import Optional, Union

from emma_datasets.datamodels.annotations import (
    ActionTrajectory,
    Caption,
    QuestionAnswerPair,
    Region,
    SceneGraph,
    TaskDescription,
)
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetModalityMap, DatasetName, MediaType
from emma_datasets.datamodels.dataset_metadata import DatasetMetadata


DatasetDict = dict[DatasetName, DatasetMetadata]


class MultiSourceInstanceMixin(BaseInstance):
    """Mixin class exposing functionalities useful for instances based on multiple datasets."""

    dataset: DatasetDict
    captions: Optional[list[Caption]]
    qa_pairs: Optional[list[QuestionAnswerPair]]
    regions: Optional[list[Region]]
    scene_graph: Optional[SceneGraph]
    trajectory: Optional[ActionTrajectory]
    task_description: Optional[list[TaskDescription]]

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
    def source_paths(self) -> Union[Path, list[Path], None]:
        """Get source paths for this instance.

        Since an instance can be mapped to more than one dataset, we assume that the source media
        is going to be identical across them. Therefore, it doesn't matter which dataset's image
        file we use since they should be identical.
        """
        return next(iter(self.dataset.values())).paths

    @property
    def features_path(self) -> Union[Path, list[Path]]:
        """Get the path to the features for this instance.

        If the instance is connected to more than one dataset, just get any one feature file.
        """
        all_feature_paths = (metadata.features_path for metadata in self.dataset.values())
        return next(all_feature_paths)

    @property
    def is_full_trajectory(self) -> bool:
        """Whether the instance corresponds to a trajectory of multiple subgoals."""
        if self.modality == MediaType.image:
            return False
        return isinstance(self.features_path, list) and len(self.features_path) > 1


class Instance(MultiSourceInstanceMixin):
    """Instance within the dataset."""

    captions: Optional[list[Caption]]
    qa_pairs: Optional[list[QuestionAnswerPair]]
    regions: Optional[list[Region]]
    scene_graph: Optional[SceneGraph]
    trajectory: Optional[ActionTrajectory]
    task_description: Optional[list[TaskDescription]]

    @property
    def language_annotations(self) -> list[str]:
        """Derives all the language annotations associated with a given instance."""
        lang_data_iterable = []

        if self.captions is not None:
            lang_data_iterable.extend([caption.get_language_data() for caption in self.captions])

        if self.qa_pairs is not None:
            lang_data_iterable.extend([qa_pair.get_language_data() for qa_pair in self.qa_pairs])

        if self.regions is not None:
            lang_data_iterable.extend([region.get_language_data() for region in self.regions])

        if self.scene_graph is not None:
            lang_data_iterable.extend(self.scene_graph.get_language_data())

        if self.trajectory is not None:
            lang_data_iterable.extend(self.trajectory.get_language_data())

        if self.task_description is not None:
            lang_data_iterable.extend([desc.get_language_data() for desc in self.task_description])

        return lang_data_iterable
