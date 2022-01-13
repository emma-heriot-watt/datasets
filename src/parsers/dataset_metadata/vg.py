from pathlib import Path
from typing import Iterator

from rich.progress import Progress

from src.datamodels import DatasetMetadata, DatasetName, MediaType, SourceMedia
from src.datamodels.datasets import VgImageMetadata
from src.io import read_json
from src.parsers.dataset_metadata.metadata_parser import DatasetMetadataParser


class VgMetadataParser(DatasetMetadataParser[VgImageMetadata]):
    """Convert VG instance metadata."""

    metadata_model = VgImageMetadata
    dataset_name = DatasetName.visual_genome

    def __init__(
        self, image_data_json_path: str, images_dir: str, regions_dir: str, progress: Progress
    ) -> None:
        super().__init__(progress=progress)
        self.image_data_json_path = image_data_json_path
        self.images_dir = images_dir
        self.regions_dir = regions_dir

    def get_metadata(self) -> Iterator[VgImageMetadata]:
        """Get all the image metadata from VG."""
        raw_data = read_json(self.image_data_json_path)
        return self.structure_raw_metadata(raw_data)

    def convert_to_dataset_metadata(self, metadata: VgImageMetadata) -> DatasetMetadata:
        """Convert single instance's metadata to the common datamodel."""
        return DatasetMetadata(
            id=metadata.image_id,
            name=self.dataset_name,
            split=metadata.dataset_split,
            media=SourceMedia(
                url=metadata.url,
                media_type=MediaType.image,
                path=Path(self.images_dir).joinpath(f"{metadata.image_id}.jpg").as_posix(),
            ),
            regions_path=Path(self.regions_dir).joinpath(f"{metadata.image_id}.json").as_posix(),
        )
