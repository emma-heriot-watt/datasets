import itertools
from collections.abc import Iterator
from typing import Any

from emma_datasets.datamodels import AnnotationType, DatasetName, QuestionAnswerPair
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


RawInstanceType = tuple[str, list[dict[str, Any]]]


class GqaQaPairExtractor(AnnotationExtractor[QuestionAnswerPair]):
    """Split QA pairs from GQA into multiple files."""

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.qa_pair

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.gqa

    def process_raw_file_return(self, raw_data: Any) -> Iterator[dict[str, Any]]:
        """Add image ID to each question data dictionary."""
        for question_id, question_data in raw_data.items():
            question_data["id"] = question_id
            yield question_data

    def postprocess_raw_data(self, raw_data: Any) -> Iterator[RawInstanceType]:
        """Group all pairs by image ID."""
        sorted_raw_data = sorted(raw_data, key=lambda qa: qa["imageId"])
        grouped_qa_pairs_generator = itertools.groupby(
            sorted_raw_data, key=lambda qa: qa["imageId"]
        )
        return (
            (image_id, list(grouped_qa_pairs))
            for image_id, grouped_qa_pairs in grouped_qa_pairs_generator
        )

    def convert(self, raw_feature: list[dict[str, Any]]) -> Iterator[QuestionAnswerPair]:
        """Convert raw instance into QA pairs."""
        return (
            QuestionAnswerPair(id=qa["id"], question=qa["question"], answer=qa["answer"])
            for qa in raw_feature
        )

    def process_single_instance(self, raw_instance: RawInstanceType) -> None:
        """Process single instance into multiple QA pairs."""
        image_id, raw_qa_pairs = raw_instance
        self._write(self.convert(raw_qa_pairs), image_id)
