from pathlib import Path
from typing import Any

from rich.progress import Progress

from emma_datasets.datamodels import (
    Annotation,
    DatasetMetadata,
    DatasetName,
    DatasetSplit,
    MediaType,
    SourceMedia,
)
from emma_datasets.datamodels.datasets import GqaImageMetadata
from emma_datasets.io import read_json
from emma_datasets.parsers.dataset_metadata.metadata_parser import DatasetMetadataParser


class GqaMetadataParser(DatasetMetadataParser[GqaImageMetadata]):
    """Parse GQA instance metadata."""

    metadata_model = GqaImageMetadata
    dataset_name = DatasetName.gqa

    def __init__(
        self,
        scene_graphs_train_path: Path,
        scene_graphs_val_path: Path,
        images_dir: Path,
        scene_graphs_dir: Path,
        qa_pairs_dir: Path,
        progress: Progress,
    ) -> None:
        super().__init__(
            progress=progress,
            data_paths=[
                (scene_graphs_train_path, DatasetSplit.train),
                (scene_graphs_val_path, DatasetSplit.valid),
            ],
        )
        self.images_dir = images_dir
        self.scene_graphs_dir = scene_graphs_dir
        self.qa_pairs_dir = qa_pairs_dir

    def convert_to_dataset_metadata(self, metadata: GqaImageMetadata) -> DatasetMetadata:
        """Convert single instance's metadata to the common datamodel."""
        return DatasetMetadata(
            id=str(metadata.id),
            name=self.dataset_name,
            split=metadata.dataset_split,
            media=SourceMedia(
                media_type=MediaType.image,
                path=self.images_dir.joinpath(metadata.file_name),
            ),
            annotation_paths={
                Annotation.scene_graph: self.scene_graphs_dir.joinpath(f"{metadata.id}.json"),
                Annotation.qa_pair: self.qa_pairs_dir.joinpath(f"{metadata.id}.json"),
            },
        )

    def _preprocess_raw_data(self, image_id: str, scene_graph: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "height": scene_graph["height"],
            "width": scene_graph["width"],
        }

    def _read(self, path: Path) -> Any:
        """Read data from the given path."""
        raw_data = read_json(path)
        return [
            self._preprocess_raw_data(image_id, scene_graph)
            for image_id, scene_graph in raw_data.items()
        ]
