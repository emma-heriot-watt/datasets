from pathlib import Path

from overrides import overrides
from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName, MediaType, SourceMedia
from emma_datasets.datamodels.datasets import ConceptualCaptionsMetadata
from emma_datasets.io import read_parquet
from emma_datasets.parsers.dataset_metadata.metadata_parser import (
    DataPathTuple,
    DatasetMetadataParser,
)


class ConceptualCaptionsMetadataParser(DatasetMetadataParser[ConceptualCaptionsMetadata]):
    """Parse Conceptual Captions."""

    metadata_model = ConceptualCaptionsMetadata
    dataset_name = DatasetName.conceptual_captions

    def __init__(
        self,
        parquet_files_dir: list[DataPathTuple],
        features_dir: list[DataPathTuple],
        captions_dir: list[DataPathTuple],
        progress: Progress,
    ) -> None:
        self.parquet_files_dir = parquet_files_dir
        self.features_dir = features_dir
        self.captions_dir = captions_dir
        self.file_ext = "parquet"
        super().__init__(data_paths=parquet_files_dir, progress=progress)

    @overrides(check_signature=False)
    def convert_to_dataset_metadata(self, metadata: ConceptualCaptionsMetadata) -> DatasetMetadata:
        """Convert a single instance of metadata model to the common DatasetMetadata."""
        split_pos = 0 if metadata.dataset_split == self.features_dir[0][1] else 1
        return DatasetMetadata(
            id=metadata.key,
            name=self.dataset_name,
            split=metadata.dataset_split,
            media=SourceMedia(
                url=metadata.url,
                media_type=MediaType.image,
                width=metadata.width,
                height=metadata.height,
            ),
            features_path=self.features_dir[split_pos][0].joinpath(
                metadata.shard_id, f"{metadata.key}.{self.feature_ext}"
            ),
            caption_path=self.captions_dir[split_pos][0].joinpath(
                metadata.shard_id, f"{metadata.key}.json"
            ),
        )

    def _get_shard_id_from_path(self, path: Path) -> str:
        return path.name.split(".")[0]

    def _read(self, path: Path) -> list[dict[str, str]]:
        """Conceptual Captions is downloaded using https://github.com/rom1504/img2dataset.

        The dataset metadata can be found inside each .parquet file for each shard. Each .parquet
        file contains the metadata for all instances associated with the shard.
        """
        metadata_shard = read_parquet(path)
        metadata_list = []
        for _, metadata in metadata_shard.iterrows():
            metadata_dict = dict(metadata)
            if metadata["status"] == "success":
                metadata_dict["shard_id"] = self._get_shard_id_from_path(path)
                metadata_list.append(metadata_dict)
        return metadata_list
