from typing import Any

from emma_datasets.datamodels import AnnotationType, DatasetName, SceneGraph
from emma_datasets.datamodels.datasets import GqaSceneGraph
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class GqaSceneGraphExtractor(AnnotationExtractor[SceneGraph]):
    """Split scene graphs from GQA into multiple files."""

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.scene_graph

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.gqa

    def process_raw_file_return(self, raw_data: Any) -> Any:
        """Get scene graph and image id as items."""
        return raw_data.items()

    def convert(self, raw_feature: GqaSceneGraph) -> SceneGraph:
        """Convert GQA scene graph into common SceneGraph."""
        return SceneGraph(
            location=raw_feature.location,
            weather=raw_feature.weather,
            objects=raw_feature.objects,
        )

    def process_single_instance(self, raw_instance: tuple[str, dict[str, Any]]) -> None:
        """Process raw scene graph and write to file."""
        image_id, raw_scene_graph = raw_instance
        raw_scene_graph["image_id"] = image_id
        gqa_scene_graph = GqaSceneGraph.parse_obj(raw_scene_graph)
        scene_graph = self.convert(gqa_scene_graph)
        self._write(scene_graph, image_id)
