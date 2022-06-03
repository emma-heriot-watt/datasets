from pathlib import Path
from typing import Any, Iterator

from overrides import overrides
from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName, MediaType, SourceMedia
from emma_datasets.datamodels.datasets import AlfredImageMetadata, AlfredMetadata
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

        self._width = 300
        self._height = 300

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
                    f"{metadata.task_id}_{metadata.scene.scene_num}_{high_idx}.{self.feature_ext}"
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
        subgoal_images = [image for image in metadata.images if image.high_idx == high_idx]
        subgoal_images = self._get_last_frame_of_low_level_action(subgoal_images)

        return [
            SourceMedia(
                media_type=MediaType.image,
                path=frames_dir.joinpath(image.image_name),
                width=self._width,
                height=self._height,
            )
            for image in subgoal_images
        ]

    def _get_last_frame_of_low_level_action(
        self, images: list[AlfredImageMetadata]
    ) -> list[AlfredImageMetadata]:
        """Keep only the last frame for each low-level action.

        ALFRED data have multiple images per low-level actions including filler frames inbetween
        low-level actions.
        """
        low_images: list[AlfredImageMetadata] = []
        prev_low_idx = -1
        for image in images[::-1]:
            if prev_low_idx != image.low_idx:
                prev_low_idx = image.low_idx
                low_images.append(image)

        return low_images[::-1]

    def _read(self, path: Path) -> Any:
        """Read JSON from the given path."""
        return [read_json(path)]
