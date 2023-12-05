from typing import Any

from emma_datasets.datamodels import AnnotationType, Caption, DatasetName
from emma_datasets.datamodels.datasets import EpicKitchensNarrationMetadata
from emma_datasets.io import read_csv
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class EpicKitchensCaptionExtractor(AnnotationExtractor[Caption]):
    """Split captions for EpicKitchens into multiple files."""

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.caption

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.epic_kitchens

    @property
    def file_ext(self) -> str:
        """The file extension of the raw data files."""
        return "csv"

    def read(self, file_path: Any) -> list[dict[str, Any]]:
        """Read Epic Kitchen CSV file."""
        return read_csv(file_path)

    def convert(self, raw_feature: EpicKitchensNarrationMetadata) -> list[Caption]:
        """Convert raw feature to caption."""
        return [Caption(text=raw_feature.narration)]

    def process_single_instance(self, raw_instance: dict[str, Any]) -> None:
        """Process raw instance and write to file."""
        structured_instance = EpicKitchensNarrationMetadata.parse_obj(raw_instance)
        caption = self.convert(structured_instance)
        self._write(caption, structured_instance.narration_id)
