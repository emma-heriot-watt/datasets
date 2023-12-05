from typing import Optional

from pydantic import parse_file_as

from emma_datasets.datamodels import (
    ActionTrajectory,
    AnnotationDatasetMap,
    AnnotationType,
    Caption,
    DatasetMetadata,
    Instance,
    QuestionAnswerPair,
    Region,
    SceneGraph,
    TaskDescription,
)
from emma_datasets.parsers.instance_creators.generic import GenericInstanceCreator


class PretrainInstanceCreator(GenericInstanceCreator[list[DatasetMetadata], Instance]):
    """Create instances from groups of metadata from all the datasets."""

    def _create_instance(self, input_data: list[DatasetMetadata]) -> Instance:
        """Create instance from a single group of metadata."""
        regions = self._get_regions(input_data)
        scene_graph = self._get_scene_graph(input_data)
        trajectory = self._get_action_trajectory(input_data)
        captions = self._get_captions(input_data)
        qa_pairs = self._get_qa_pairs(input_data)
        task_description = self._get_task_description(input_data)

        return Instance(
            dataset={metadata.name: metadata for metadata in input_data},
            captions=captions,
            qa_pairs=qa_pairs,
            regions=regions,
            scene_graph=scene_graph,
            trajectory=trajectory,
            task_description=task_description,
        )

    def _get_regions(self, metadata_list: list[DatasetMetadata]) -> Optional[list[Region]]:
        """Get regions for instance from given path in dataset metadata."""
        filtered_metadata_list = self._filter_metadata_list(metadata_list, AnnotationType.region)

        if not filtered_metadata_list:
            return None

        # If it is not None, we are assuming there is ONLY one in the list.
        metadata = filtered_metadata_list[0]

        if metadata.regions_path is None:
            raise ValueError("`metadata.regions_path` should not be `None`")

        return parse_file_as(list[Region], metadata.regions_path)

    def _get_scene_graph(self, metadata_list: list[DatasetMetadata]) -> Optional[SceneGraph]:
        """Get scene graph for scene from given path."""
        filtered_metadata_list = self._filter_metadata_list(
            metadata_list, AnnotationType.scene_graph
        )

        if not filtered_metadata_list:
            return None

        # If it is not None, we are assuming there is ONLY one in the list.
        metadata = filtered_metadata_list[0]

        if metadata.scene_graph_path is None:
            raise ValueError("`metadata.scene_graph_path` should not be `None`")

        return SceneGraph.parse_file(metadata.scene_graph_path)

    def _get_action_trajectory(
        self, metadata_list: list[DatasetMetadata]
    ) -> Optional[ActionTrajectory]:
        filtered_metadata_list = self._filter_metadata_list(
            metadata_list, AnnotationType.action_trajectory
        )

        if not filtered_metadata_list:
            return None

        # If not None, assume only ONE trajectory in the list
        metadata = filtered_metadata_list[0]

        if metadata.action_trajectory_path is None:
            raise ValueError("`metadata.action_trajectory_path` should not be `None`")

        return ActionTrajectory.parse_file(metadata.action_trajectory_path)

    def _get_task_description(
        self, metadata_list: list[DatasetMetadata]
    ) -> Optional[list[TaskDescription]]:
        filtered_metadata_list = self._filter_metadata_list(
            metadata_list, AnnotationType.task_description
        )

        if not filtered_metadata_list:
            return None

        # If not None, assume only ONE trajectory in the list
        metadata = filtered_metadata_list[0]
        if metadata.task_description_path is None:
            raise ValueError("`metadata.task_description_path` should not be `None`")

        return parse_file_as(list[TaskDescription], metadata.task_description_path)

    def _get_captions(self, metadata_list: list[DatasetMetadata]) -> list[Caption]:
        """Get captions for instance."""
        filtered_metadata_list = self._filter_metadata_list(metadata_list, AnnotationType.caption)

        if not filtered_metadata_list:
            return []

        captions = []

        for metadata in filtered_metadata_list:
            if metadata.caption_path is None:
                raise ValueError("`metadata.caption_path` should not be `None`")

            captions.extend(parse_file_as(list[Caption], metadata.caption_path))

        return captions

    def _get_qa_pairs(self, metadata_list: list[DatasetMetadata]) -> list[QuestionAnswerPair]:
        """Get question answer pairs for instance."""
        filtered_metadata_list = self._filter_metadata_list(metadata_list, AnnotationType.qa_pair)

        if not filtered_metadata_list:
            return []

        qa_pairs = []

        for metadata in filtered_metadata_list:
            if metadata.qa_pairs_path is None:
                raise ValueError("`metadata.qa_pairs_path` should not be `None`")

            try:
                qa_pairs.extend(parse_file_as(list[QuestionAnswerPair], metadata.qa_pairs_path))
            except FileNotFoundError:
                # TODO(amit): add reasoning for this exception in docstring
                pass  # noqa: WPS420

        return qa_pairs

    def _filter_metadata_list(
        self, metadata_list: list[DatasetMetadata], annotation: AnnotationType
    ) -> list[DatasetMetadata]:
        return [
            metadata
            for metadata in metadata_list
            if metadata.name in AnnotationDatasetMap[annotation]
        ]
