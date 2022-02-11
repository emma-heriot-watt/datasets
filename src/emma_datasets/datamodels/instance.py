import re
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


def _get_language_data_from_scene_graph(scene_graph: SceneGraph) -> list[str]:
    annotations = []

    for _, scene_obj in scene_graph.objects.items():
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


def _get_language_data_from_caption(caption: Caption) -> list[str]:
    """Returns the caption text."""
    return [caption.text]


def _get_language_data_from_qa(qa: QuestionAnswerPair) -> list[str]:
    """Returns a formatted string containing both the question and the answer."""
    return [f"{qa.question} {qa.answer}"]


def _get_language_data_from_regions(regions: list[Region]) -> list[str]:
    """Returns the region descriptions for each region of the image."""
    return [region.caption for region in regions]


AnnotationType = Union[Caption, SceneGraph, list[Region], QuestionAnswerPair, ActionTrajectory]


def get_language_data(attribute_value: AnnotationType) -> list[str]:
    """Returns the language annotations associated with a given attribute."""
    if isinstance(attribute_value, Caption):
        return _get_language_data_from_caption(attribute_value)

    if isinstance(attribute_value, QuestionAnswerPair):
        return _get_language_data_from_qa(attribute_value)

    if isinstance(attribute_value, list) and isinstance(attribute_value[0], Region):
        return _get_language_data_from_regions(attribute_value)

    if isinstance(attribute_value, SceneGraph):
        return _get_language_data_from_scene_graph(attribute_value)

    return _get_language_data_from_trajectory(attribute_value)


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
    def language_annotations(self) -> list[str]:
        """Derives all the language annotations associated with a given instance."""
        lang_data_iterable = []

        # iterates over the attributes of the class
        for attribute_name, attribute_value in self:
            # if the current attribute is available and it's not the `dataset` name
            if attribute_name != "dataset" and attribute_value is not None:
                lang_data = get_language_data(attribute_value)
                lang_data_iterable.extend(lang_data)

        return lang_data_iterable

    def paths(self) -> Union[Path, list[Path], None]:
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
