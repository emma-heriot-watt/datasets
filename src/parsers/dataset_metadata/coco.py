import itertools
from pathlib import Path
from typing import Iterator

from rich.progress import Progress

from src.datamodels import DatasetMetadata, DatasetName, DatasetSplit, MediaType, SourceMedia
from src.datamodels.datasets import CocoImageMetadata
from src.io import read_json
from src.parsers.dataset_metadata.metadata_parser import DatasetMetadataParser


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
        super().__init__(progress=progress)
        self.caption_train_path = caption_train_path
        self.caption_val_path = caption_val_path
        self.images_dir = images_dir
        self.captions_dir = captions_dir

    def get_metadata(self) -> Iterator[CocoImageMetadata]:
        """Get all the image metadat from COCO."""
        train_data = read_json(self.caption_train_path)["images"]
        val_data = read_json(self.caption_val_path)["images"]

        structured_train = self.structure_raw_metadata(train_data, DatasetSplit.train)
        structured_val = self.structure_raw_metadata(val_data, DatasetSplit.valid)

        return itertools.chain.from_iterable([structured_train, structured_val])

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
