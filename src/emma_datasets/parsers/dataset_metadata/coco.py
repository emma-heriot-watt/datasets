from pathlib import Path
from typing import Any

from rich.progress import Progress

from emma_datasets.datamodels import (
    DatasetMetadata,
    DatasetName,
    DatasetSplit,
    MediaType,
    SourceMedia,
)
from emma_datasets.datamodels.datasets import CocoImageMetadata
from emma_datasets.io import read_json
from emma_datasets.parsers.dataset_metadata.metadata_parser import DatasetMetadataParser


class CocoMetadataParser(DatasetMetadataParser[CocoImageMetadata]):
    """Parse instance metadata for COCO."""

    metadata_model = CocoImageMetadata
    dataset_name = DatasetName.coco

    def __init__(
        self,
        caption_train_path: Path,
        caption_val_path: Path,
        images_dir: Path,
        captions_dir: Path,
        progress: Progress,
    ) -> None:
        super().__init__(
            progress=progress,
            data_paths=[
                (caption_train_path, DatasetSplit.train),
                (caption_val_path, DatasetSplit.valid),
            ],
        )

        self.images_dir = images_dir
        self.captions_dir = captions_dir

    def convert_to_dataset_metadata(self, metadata: CocoImageMetadata) -> DatasetMetadata:
        """Convert single instance's metadata to the common datamodel."""
        return DatasetMetadata(
            id=str(metadata.id),
            name=self.dataset_name,
            split=metadata.dataset_split,
            media=SourceMedia(
                url=metadata.coco_url,
                media_type=MediaType.image,
                path=self.images_dir.joinpath(metadata.file_name).as_posix(),
            ),
            caption_path=self.captions_dir.joinpath(f"{metadata.id}.json").as_posix(),
        )

    def _read(self, path: Path) -> Any:
        """Read data from the given path."""
        return read_json(path)["images"]
