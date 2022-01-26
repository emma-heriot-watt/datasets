from pathlib import Path
from typing import Any

from rich.progress import Progress

from emma_datasets.datamodels import (
    Annotation,
    DatasetMetadata,
    DatasetName,
    MediaType,
    SourceMedia,
)
from emma_datasets.datamodels.datasets import EpicKitchensNarrationMetadata
from emma_datasets.io import read_csv
from emma_datasets.parsers.dataset_metadata.metadata_parser import (
    DataPathTuple,
    DatasetMetadataParser,
)


class EpicKitchensMetadataParser(DatasetMetadataParser[EpicKitchensNarrationMetadata]):
    """EPIC-KITCHENS Metadata Parser."""

    metadata_model = EpicKitchensNarrationMetadata
    dataset_name = DatasetName.epic_kitchens
    file_ext = "csv"

    def __init__(
        self,
        data_paths: list[DataPathTuple],
        frames_dir: Path,
        captions_dir: Path,
        progress: Progress,
    ) -> None:
        super().__init__(data_paths=data_paths, progress=progress)

        self.frames_dir = frames_dir
        self.captions_dir = captions_dir

    def convert_to_dataset_metadata(
        self, metadata: EpicKitchensNarrationMetadata
    ) -> DatasetMetadata:
        """Convert Narration metadata to DatasetMetadata."""
        return DatasetMetadata(
            id=metadata.narration_id,
            name=self.dataset_name,
            split=metadata.dataset_split,
            media=self.get_all_source_media_for_narration(metadata),
            annotation_paths={
                Annotation.caption: self.captions_dir.joinpath(f"{metadata.narration_id}.json")
            },
        )

    def get_all_source_media_for_narration(
        self, metadata: EpicKitchensNarrationMetadata
    ) -> list[SourceMedia]:
        """Get all the source media for a given subgoal from EPIC-KITCHENS.

        A big assumption is made that all the RGB frames are separated per video, within different subdirectories for that video.

        Each frame is named like `frame_0000028665.jpg`: a total of 10 digits.
        """
        max_padding_length = 10

        frames_for_video_dir = self.frames_dir.joinpath(f"{metadata.video_id}/")

        return [
            SourceMedia(
                media_type=MediaType.image,
                path=frames_for_video_dir.joinpath(
                    f"frame_{str(frame_number).zfill(max_padding_length)}.jpg"
                ),
            )
            for frame_number in range(metadata.start_frame, metadata.stop_frame + 1)
        ]

    def _read(self, path: Path) -> Any:
        """Read CSV data from the given path."""
        return read_csv(path)
