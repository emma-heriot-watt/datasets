from pathlib import Path
from typing import Any

from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName, MediaType, SourceMedia
from emma_datasets.datamodels.datasets import VgImageMetadata
from emma_datasets.io import read_json
from emma_datasets.parsers.dataset_metadata.metadata_parser import DatasetMetadataParser


class VgMetadataParser(DatasetMetadataParser[VgImageMetadata]):
    """Convert VG instance metadata."""

    metadata_model = VgImageMetadata
    dataset_name = DatasetName.visual_genome

    def __init__(
        self,
        image_data_json_path: Path,
        images_dir: Path,
        regions_dir: Path,
        progress: Progress,
    ) -> None:
        super().__init__(progress=progress, data_paths=[(image_data_json_path, None)])

        self.images_dir = images_dir
        self.regions_dir = regions_dir

    def convert_to_dataset_metadata(self, metadata: VgImageMetadata) -> DatasetMetadata:
        """Convert single instance's metadata to the common datamodel."""
        return DatasetMetadata(
            id=metadata.image_id,
            name=self.dataset_name,
            split=metadata.dataset_split,
            media=SourceMedia(
                url=metadata.url,
                media_type=MediaType.image,
                path=self.images_dir.joinpath(f"{metadata.image_id}.jpg").as_posix(),
            ),
            regions_path=self.regions_dir.joinpath(f"{metadata.image_id}.json").as_posix(),
        )

    def _read(self, path: Path) -> Any:
        """Read data from the given path."""
        return read_json(path)
