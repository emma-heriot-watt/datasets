import re
from pathlib import Path
from typing import Optional, Union

from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetModalityMap, DatasetName, MediaType
from emma_datasets.datamodels.dataset_metadata import DatasetMetadata
from emma_datasets.datamodels.datasets import AlfredHighAction, AlfredLowAction
from emma_datasets.datamodels.region import Region
from emma_datasets.datamodels.scene_graph import SceneGraph
from emma_datasets.datamodels.text import Caption, QuestionAnswerPair
from emma_datasets.datamodels.trajectory import GenericActionTrajectory


DatasetDict = dict[DatasetName, DatasetMetadata]

ActionTrajectory = GenericActionTrajectory[AlfredLowAction, AlfredHighAction]


def _get_language_data_from_scene_graph(scene_graph: SceneGraph) -> list[str]:
    annotations = []

    for scene_obj in scene_graph.objects.values():
        if scene_obj.attributes:
            for attr in scene_obj.attributes:
                annotations.append(f"{scene_obj.name} has attribute {attr}")

        if scene_obj.relations:
            for rel in scene_obj.relations:
                rel_object = scene_graph.objects[rel.object]
                annotations.append(f"{scene_obj.name} {rel.name} {rel_object.name}")

    return annotations


def get_action_string(action_name: str) -> str:
    """Returns a phrase associated with the action API name.

    API action names are in camelcase format: MoveAhead_25
    """
    parts: list[str] = []

    for x in re.findall("[A-Z][^A-Z]*", action_name):
        parts.extend(xi for xi in x.split("_"))

    return " ".join(parts)


def _get_language_data_from_trajectory(trajectory: ActionTrajectory) -> list[str]:
    trajectory_str = " ".join(
        get_action_string(low_action.discrete_action.action)
        for low_action in trajectory.low_level_actions
    )
    return [trajectory_str]


def _get_language_data_from_captions(captions: list[Caption]) -> list[str]:
    """Returns the caption text."""
    return [caption.text for caption in captions]


def _get_language_data_from_qas(qas: list[QuestionAnswerPair]) -> list[str]:
    """Returns a formatted string containing both the question and the answer."""
    return [f"{qa.question} {qa.answer}" for qa in qas]


def _get_language_data_from_regions(regions: list[Region]) -> list[str]:
    """Returns the region descriptions for each region of the image."""
    return [region.caption for region in regions]


class Instance(BaseInstance):
    """Instance within the dataset."""

    dataset: DatasetDict
    captions: Optional[list[Caption]]
    qas: Optional[list[QuestionAnswerPair]]
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
    def language_annotations(self) -> list[str]:
        """Derives all the language annotations associated with a given instance."""
        lang_data_iterable = []

        if self.captions is not None:
            lang_data_iterable.extend(_get_language_data_from_captions(self.captions))

        if self.qas is not None:
            lang_data_iterable.extend(_get_language_data_from_qas(self.qas))

        if self.regions is not None:
            lang_data_iterable.extend(_get_language_data_from_regions(self.regions))

        if self.scene_graph is not None:
            lang_data_iterable.extend(_get_language_data_from_scene_graph(self.scene_graph))

        if self.trajectory is not None:
            lang_data_iterable.extend(_get_language_data_from_trajectory(self.trajectory))

        return lang_data_iterable

    @property
    def source_paths(self) -> Union[Path, list[Path], None]:
        """Get source paths for this instance.

        Since an instance can be mapped to more than one dataset, we assume that the source media
        is going to be identical across them. Therefore, it doesn't matter which dataset's image
        file we use since they should be identical.
        """
        return next(iter(self.dataset.values())).paths

    @property
    def features_path(self) -> Path:
        """Get the path to the features for this instance.

        If the instance is connected to more than one dataset, just get any one feature file.
        """
        all_feature_paths = (metadata.features_path for metadata in self.dataset.values())
        return next(all_feature_paths)
