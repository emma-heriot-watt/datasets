import itertools
from pathlib import Path
from typing import Any, Iterator

from rich.progress import Progress

from src.datamodels import DatasetMetadata, DatasetName, DatasetSplit, MediaType, SourceMedia
from src.datamodels.datasets import GqaImageMetadata
from src.io import read_json
from src.parsers.dataset_metadata.metadata_parser import DatasetMetadataParser


class GqaMetadataParser(DatasetMetadataParser[GqaImageMetadata]):
    """Parse GQA instance metadata."""

    metadata_model = GqaImageMetadata
    dataset_name = DatasetName.gqa

    def __init__(
        self,
        scene_graphs_train_path: str,
        scene_graphs_val_path: str,
        images_dir: str,
        scene_graphs_dir: str,
        qa_pairs_dir: str,
        progress: Progress,
    ) -> None:
        super().__init__(progress=progress)
        self.scene_graphs_train_path = scene_graphs_train_path
        self.scene_graphs_val_path = scene_graphs_val_path
        self.images_dir = images_dir
        self.scene_graphs_dir = scene_graphs_dir
        self.qa_pairs_dir = qa_pairs_dir

    def get_metadata(self) -> Iterator[GqaImageMetadata]:
        """Get all the instance metadata for GQA."""
        train_data = read_json(self.scene_graphs_train_path)
        val_data = read_json(self.scene_graphs_val_path)

        train_data = [
            self._preprocess_raw_data(image_id, scene_graph)
            for image_id, scene_graph in train_data.items()
        ]
        val_data = [
            self._preprocess_raw_data(image_id, scene_graph)
            for image_id, scene_graph in val_data.items()
        ]

        structured_train = self.structure_raw_metadata(train_data, DatasetSplit.train)
        structured_val = self.structure_raw_metadata(val_data, DatasetSplit.valid)

        return itertools.chain.from_iterable([structured_train, structured_val])

    def convert_to_dataset_metadata(self, metadata: GqaImageMetadata) -> DatasetMetadata:
        """Convert single instance's metadata to the common datamodel."""
        return DatasetMetadata(
            id=str(metadata.id),
            name=self.dataset_name,
            split=metadata.dataset_split,
            media=SourceMedia(
                media_type=MediaType.image,
                path=Path(self.images_dir).joinpath(metadata.file_name).as_posix(),
            ),
            scene_graph_path=Path(self.scene_graphs_dir)
            .joinpath(f"{metadata.id}.json")
            .as_posix(),
            qa_pairs_path=Path(self.qa_pairs_dir).joinpath(f"{metadata.id}.json").as_posix(),
        )

    def _preprocess_raw_data(self, image_id: str, scene_graph: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "height": scene_graph["height"],
            "width": scene_graph["width"],
        }
