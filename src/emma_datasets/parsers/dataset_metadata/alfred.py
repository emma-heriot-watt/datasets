from pathlib import Path
from typing import Any, Iterator

from overrides import overrides
from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName, MediaType, SourceMedia
from emma_datasets.datamodels.datasets import AlfredMetadata
from emma_datasets.io import read_json
from emma_datasets.parsers.dataset_metadata.metadata_parser import (
    DataPathTuple,
    DatasetMetadataParser,
)


class AlfredMetadataParser(DatasetMetadataParser[AlfredMetadata]):
    """Parse ALFRED metadata into subgoals."""

    metadata_model = AlfredMetadata
    dataset_name = DatasetName.alfred

    def __init__(
        self,
        data_paths: list[DataPathTuple],
        progress: Progress,
        alfred_dir: Path,
        captions_dir: Path,
        trajectories_dir: Path,
        features_dir: Path,
    ) -> None:
        super().__init__(data_paths=data_paths, progress=progress)

        self.alfred_dir = alfred_dir
        self.captions_dir = captions_dir
        self.trajectories_dir = trajectories_dir
        self.features_dir = features_dir

    @overrides(check_signature=False)
    def convert_to_dataset_metadata(self, metadata: AlfredMetadata) -> Iterator[DatasetMetadata]:
        """Convert ALFRED metadata to DatasetMetadata."""
        num_subgoals = min(len(ann.high_descs) for ann in metadata.turk_annotations["anns"])

        path_to_frames = self.alfred_dir.glob(
            f"*/{metadata.task_type}-*-{metadata.scene.scene_num}/{metadata.task_id}/"
        )
        frames_dir = next(iter(path_to_frames)).joinpath("raw_images")

        for high_idx in range(num_subgoals):  # noqa: WPS526
            yield DatasetMetadata(
                id=metadata.task_id,
                name=self.dataset_name,
                split=metadata.dataset_split,
                media=self.get_all_source_media_for_subgoal(metadata, frames_dir, high_idx),
                features_path=self.features_dir.joinpath(
                    f"{metadata.task_id}_{high_idx}.{self.feature_ext}"
                ),
                caption_path=self.captions_dir.joinpath(f"{metadata.task_id}_{high_idx}.json"),
                action_trajectory_path=self.trajectories_dir.joinpath(
                    f"{metadata.task_id}_{high_idx}.json"
                ),
            )

    def get_all_source_media_for_subgoal(
        self, metadata: AlfredMetadata, frames_dir: Path, high_idx: int
    ) -> list[SourceMedia]:
        """Get all images for the given subgoal."""
        return [
            SourceMedia(
                media_type=MediaType.image,
                path=frames_dir.joinpath(image.image_name),
            )
            for image in metadata.images
            if image.high_idx == high_idx
        ]

    def _read(self, path: Path) -> Any:
        """Read JSON from the given path."""
        return [read_json(path)]
