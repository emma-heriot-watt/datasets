import itertools
from multiprocessing.pool import Pool
from typing import Iterable, Iterator, Optional, TypeVar

from pydantic import parse_file_as
from rich.progress import Progress

from src.datamodels import (
    Caption,
    DatasetName,
    Instance,
    QuestionAnswerPair,
    Region,
    Scene,
    SceneGraph,
)


T = TypeVar("T")


class StructureInstances:
    """Structure instances from scenes."""

    def __init__(self, progress: Progress) -> None:
        self.task_id = progress.add_task(
            "Creating instances", visible=False, start=False, total=float("inf")
        )

    def from_scenes(
        self, scenes: Iterable[Scene], progress: Progress, pool: Optional[Pool]
    ) -> Iterator[Instance]:
        """Get all instances from all scenes."""
        progress.reset(self.task_id, start=True, visible=True)

        if pool is not None:
            for instances in pool.imap_unordered(self.get_instances, scenes):
                progress.advance(self.task_id, advance=len(instances))
                yield from itertools.chain(instances)

        else:
            for scene in scenes:
                scene_instances = self.get_instances(scene)
                progress.advance(self.task_id, advance=len(scene_instances))
                yield from itertools.chain(scene_instances)

    def get_instances(self, scene: Scene) -> list[Instance]:
        """Get all the instances from a single scene."""
        regions = self._get_regions(scene)
        scene_graph = self._get_scene_graph(scene)
        captions = self._get_captions(scene)
        qa_pairs = self._get_qa_pairs(scene)

        caption_instances = (
            self._instances_from_captions(scene, captions, regions, scene_graph)
            if captions
            else None
        )

        qa_pair_instances = (
            self._instances_from_qa_pairs(scene, qa_pairs, regions, scene_graph)
            if qa_pairs
            else None
        )

        instance_iterators = [
            iterator for iterator in (caption_instances, qa_pair_instances) if iterator is not None
        ]

        # If there are no instances with text, return without
        if not instance_iterators:
            return [self._instance_from_textless_scene(scene, scene_graph, regions)]

        return list(itertools.chain.from_iterable(instance_iterators))

    def _instances_from_qa_pairs(
        self,
        scene: Scene,
        qa_pairs: list[QuestionAnswerPair],
        regions: Optional[list[Region]],
        scene_graph: Optional[SceneGraph],
    ) -> Iterator[Instance]:
        return (
            Instance(
                media_type=scene.media_type,
                dataset=scene.dataset,
                qa=qa_pair,
                regions=regions,
                scene_graph=scene_graph,
            )
            for qa_pair in qa_pairs
        )

    def _instances_from_captions(
        self,
        scene: Scene,
        captions: list[Caption],
        regions: Optional[list[Region]],
        scene_graph: Optional[SceneGraph],
    ) -> Iterator[Instance]:
        """Get all instances from a list of captions."""
        return (
            Instance(
                media_type=scene.media_type,
                dataset=scene.dataset,
                caption=text,
                regions=regions,
                scene_graph=scene_graph,
            )
            for text in captions
        )

    def _instance_from_textless_scene(
        self, scene: Scene, scene_graph: Optional[SceneGraph], regions: Optional[list[Region]]
    ) -> Instance:
        """Return instance from a scene without any text."""
        if not regions or not len(regions):
            raise ValueError(
                "There are [b]no captions nor any QA pairs nor regions[/] for the current scene. Is this right?",
            )

        return Instance(
            media_type=scene.media_type,
            dataset=scene.dataset,
            regions=regions,
            scene_graph=scene_graph,
        )

    def _get_regions(
        self, scene: Scene, dataset_name: DatasetName = DatasetName.visual_genome
    ) -> Optional[list[Region]]:
        """Get regions for instance from given path in dataset metadata."""
        metadata = scene.dataset.get(dataset_name, None)

        if metadata is None:
            return None

        if metadata.regions_path is None:
            raise ValueError("`metadata.regions_path` should not be `None`")

        return parse_file_as(list[Region], metadata.regions_path)

    def _get_scene_graph(
        self, scene: Scene, dataset_name: DatasetName = DatasetName.gqa
    ) -> Optional[SceneGraph]:
        """Get scene graph for scene from given path."""
        metadata = scene.dataset.get(dataset_name, None)

        if metadata is None:
            return None

        if metadata.scene_graph_path is None:
            raise ValueError("`metadata.scene_graph_path` should not be `None`")

        return SceneGraph.parse_file(metadata.scene_graph_path)

    def _get_captions(
        self, scene: Scene, dataset_name: DatasetName = DatasetName.coco
    ) -> list[Caption]:
        """Get captions for instance."""
        metadata = scene.dataset.get(dataset_name, None)

        if metadata is None:
            return []

        if metadata.caption_path is None:
            raise ValueError("`metadata.caption_path` should not be `None`")

        return parse_file_as(list[Caption], metadata.caption_path)

    def _get_qa_pairs(
        self, scene: Scene, dataset_name: DatasetName = DatasetName.gqa
    ) -> list[QuestionAnswerPair]:
        """Get question answer pairs for instance."""
        metadata = scene.dataset.get(dataset_name, None)

        if metadata is None:
            return []

        if metadata.qa_pairs_path is None:
            raise ValueError("`metadata.qa_pairs_path` should not be `None`")

        try:
            return parse_file_as(list[QuestionAnswerPair], metadata.qa_pairs_path)
        except FileNotFoundError:
            return []
