import itertools
from multiprocessing.pool import Pool
from typing import Iterable, Iterator, Optional

from pydantic import parse_file_as
from rich.progress import Progress

from emma_datasets.datamodels import (
    ActionTrajectory,
    AlfredHighAction,
    AlfredLowAction,
    Annotation,
    AnnotationDatasetMap,
    Caption,
    DatasetMetadata,
    GenericActionTrajectory,
    Instance,
    QuestionAnswerPair,
    Region,
    SceneGraph,
)


class InstanceCreator:
    """Create instances from groups of metadata from all the datasets."""

    def __init__(self, progress: Progress) -> None:
        self.task_id = progress.add_task(
            "Creating instances",
            visible=False,
            start=False,
            total=float("inf"),
            comment="",
        )

    def __call__(
        self,
        grouped_metadata: Iterable[list[DatasetMetadata]],
        progress: Progress,
        pool: Optional[Pool] = None,
    ) -> Iterator[Instance]:
        """Create instances from list of groups of metadata."""
        progress.reset(self.task_id, start=True, visible=True)

        if pool is not None:
            iterator = pool.imap_unordered(self.create_instances_from_metadata, grouped_metadata)
            for instances in iterator:
                progress.advance(self.task_id, advance=len(instances))
                yield from itertools.chain(instances)

        else:
            for scene in grouped_metadata:
                scene_instances = self.create_instances_from_metadata(scene)
                progress.advance(self.task_id, advance=len(scene_instances))
                yield from itertools.chain(scene_instances)

    def create_instances_from_metadata(
        self, metadata_group: list[DatasetMetadata]
    ) -> list[Instance]:
        """Create all the possible instances from a single group of metadata.

        If there are no text aspects (e.g. captions or qa pairs), then the instance is returned
        without one.
        """
        regions = self._get_regions(metadata_group)
        scene_graph = self._get_scene_graph(metadata_group)
        trajectory = self._get_action_trajectory(metadata_group)
        captions = self._get_captions(metadata_group)
        qa_pairs = self._get_qa_pairs(metadata_group)

        caption_instances = (
            self._instances_from_captions(
                metadata_group, captions, regions, scene_graph, trajectory
            )
            if captions
            else None
        )

        qa_pair_instances = (
            self._instances_from_qa_pairs(
                metadata_group, qa_pairs, regions, scene_graph, trajectory
            )
            if qa_pairs
            else None
        )

        instance_iterators = [
            iterator for iterator in (caption_instances, qa_pair_instances) if iterator is not None
        ]

        # If there are no instances with text, return without
        if not instance_iterators:
            return [self._instance_without_text(metadata_group, scene_graph, regions, trajectory)]

        return list(itertools.chain.from_iterable(instance_iterators))

    def _instances_from_qa_pairs(
        self,
        metadata_list: list[DatasetMetadata],
        qa_pairs: list[QuestionAnswerPair],
        regions: Optional[list[Region]],
        scene_graph: Optional[SceneGraph],
        trajectory: Optional[ActionTrajectory],
    ) -> Iterator[Instance]:
        """Create instances from provided QA-pairs."""
        return (
            Instance(
                dataset={metadata.name: metadata for metadata in metadata_list},
                qa=qa_pair,
                regions=regions,
                scene_graph=scene_graph,
                trajectory=trajectory,
            )
            for qa_pair in qa_pairs
        )

    def _instances_from_captions(
        self,
        metadata_list: list[DatasetMetadata],
        captions: list[Caption],
        regions: Optional[list[Region]],
        scene_graph: Optional[SceneGraph],
        trajectory: Optional[ActionTrajectory],
    ) -> Iterator[Instance]:
        """Get all instances from a list of captions."""
        return (
            Instance(
                dataset={metadata.name: metadata for metadata in metadata_list},
                caption=text,
                regions=regions,
                scene_graph=scene_graph,
                trajectory=trajectory,
            )
            for text in captions
        )

    def _instance_without_text(
        self,
        metadata_list: list[DatasetMetadata],
        scene_graph: Optional[SceneGraph],
        regions: Optional[list[Region]],
        trajectory: Optional[ActionTrajectory],
    ) -> Instance:
        """Return instance from a scene without any text."""
        if not regions or not len(regions):
            raise ValueError(
                "There are [b]no captions nor any QA pairs nor regions[/] for the current scene. Is this right?",
            )

        return Instance(
            dataset={metadata.name: metadata for metadata in metadata_list},
            regions=regions,
            scene_graph=scene_graph,
            trajectory=trajectory,
        )

    def _get_regions(self, metadata_list: list[DatasetMetadata]) -> Optional[list[Region]]:
        """Get regions for instance from given path in dataset metadata."""
        filtered_metadata_list = self._filter_metadata_list(metadata_list, Annotation.region)

        if not filtered_metadata_list:
            return None

        # If it is not None, we are assuming there is ONLY one in the list.
        metadata = filtered_metadata_list[0]

        if metadata.regions_path is None:
            raise ValueError("`metadata.regions_path` should not be `None`")

        return parse_file_as(list[Region], metadata.regions_path)

    def _get_scene_graph(self, metadata_list: list[DatasetMetadata]) -> Optional[SceneGraph]:
        """Get scene graph for scene from given path."""
        filtered_metadata_list = self._filter_metadata_list(metadata_list, Annotation.scene_graph)

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
            metadata_list, Annotation.action_trajectory
        )

        if not filtered_metadata_list:
            return None

        # If not None, assume only ONE trajectory in the list
        metadata = filtered_metadata_list[0]

        if metadata.action_trajectory_path is None:
            raise ValueError("`metadata.action_trajectory_path` should not be `None`")

        return GenericActionTrajectory[AlfredLowAction, AlfredHighAction].parse_file(
            metadata.action_trajectory_path
        )

    def _get_captions(self, metadata_list: list[DatasetMetadata]) -> list[Caption]:
        """Get captions for instance."""
        filtered_metadata_list = self._filter_metadata_list(metadata_list, Annotation.caption)

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
        filtered_metadata_list = self._filter_metadata_list(metadata_list, Annotation.qa_pair)

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
        self, metadata_list: list[DatasetMetadata], annotation: Annotation
    ) -> list[DatasetMetadata]:
        return [
            metadata
            for metadata in metadata_list
            if metadata.name in AnnotationDatasetMap[annotation]
        ]
