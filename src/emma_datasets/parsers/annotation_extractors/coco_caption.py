import itertools
from collections.abc import Iterator
from typing import Any

from pydantic import parse_obj_as

from emma_datasets.datamodels import AnnotationType, Caption, DatasetName
from emma_datasets.datamodels.datasets import CocoCaption
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class CocoCaptionExtractor(AnnotationExtractor[Caption]):
    """Split COCO captions into multiple files."""

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.caption

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.coco

    def process_raw_file_return(self, raw_data: Any) -> Any:
        """Only get the captions from the raw file."""
        return raw_data["annotations"]

    def postprocess_raw_data(self, raw_data: Any) -> Any:
        """Group the captions by image ID."""
        sorted_raw_data = sorted(raw_data, key=lambda k: k["image_id"])
        grouped_captions_generator = itertools.groupby(
            sorted_raw_data, key=lambda k: k["image_id"]
        )
        return (
            (image_id, list(grouped_captions))
            for image_id, grouped_captions in grouped_captions_generator
        )

    def convert(self, raw_feature: list[CocoCaption]) -> Iterator[Caption]:
        """Convert objects to the common Caption."""
        return (Caption(text=instance.caption) for instance in raw_feature)

    def process_single_instance(self, raw_instance: Any) -> None:
        """Process raw instance and write to file."""
        image_id, grouped_captions = raw_instance
        structured_raw = parse_obj_as(list[CocoCaption], grouped_captions)
        features = self.convert(structured_raw)
        self._write(features, image_id)
