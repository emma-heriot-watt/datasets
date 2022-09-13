import itertools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from overrides import overrides

from emma_datasets.datamodels import AnnotationType, DatasetName, QuestionAnswerPair
from emma_datasets.datamodels.datasets.utils.vqa_v2_utils import normalize_answer
from emma_datasets.datamodels.datasets.vqa_v2 import read_vqa_v2_json
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


RawInstanceType = tuple[str, list[dict[str, Any]]]


class VQAv2QaPairExtractor(AnnotationExtractor[QuestionAnswerPair]):
    """Split QA pairs from VQA-v2 into multiple files."""

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.qa_pair

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.vqa_v2

    def process_raw_file_return(self, raw_data: Any) -> Iterator[dict[str, Any]]:
        """Add image ID to each question data dictionary."""
        for question_id, question_data in raw_data.items():
            question_data["id"] = question_id
            yield question_data

    def postprocess_raw_data(self, raw_data: Any) -> Iterator[RawInstanceType]:
        """Group all pairs by image ID."""
        sorted_raw_data = sorted(raw_data, key=lambda qa: qa["image_id"])
        grouped_qa_pairs_generator = itertools.groupby(
            sorted_raw_data, key=lambda qa: qa["image_id"]
        )
        return (
            (image_id, list(grouped_qa_pairs))
            for image_id, grouped_qa_pairs in grouped_qa_pairs_generator
        )

    def convert(self, raw_feature: list[dict[str, Any]]) -> Iterator[QuestionAnswerPair]:
        """Convert raw instance into QA pairs."""
        return (
            QuestionAnswerPair(
                id=qa["id"],
                question=qa["question"],
                answer=self._get_vqa_v2_answers(qa),
            )
            for qa in raw_feature
        )

    def process_single_instance(self, raw_instance: RawInstanceType) -> None:
        """Process single instance into multiple QA pairs."""
        image_id, raw_qa_pairs = raw_instance
        self._write(self.convert(raw_qa_pairs), f"vqa_v2_{image_id}")

    @overrides(check_signature=False)
    def read(self, file_path: tuple[Path, Path]) -> Any:
        """Read the annotations."""
        if not (file_path[0].exists() and file_path[1].exists()):
            raise FileNotFoundError(f"Files {file_path} do not exist.")

        questions = read_vqa_v2_json(file_path[0], "questions")
        answers = read_vqa_v2_json(file_path[1], "annotations")
        for question_id, data in questions.items():
            data.update(answers[question_id])

        return questions

    def _get_vqa_v2_answers(self, instance: dict[str, Any]) -> list[str]:
        """Get all vqa_v2 answers."""
        return [normalize_answer(answer["answer"]) for answer in instance["answers"]]

    def _get_all_file_paths(self) -> None:
        """Get all the file paths for the dataset and store in state."""
        self.file_paths = self._paths  # type: ignore[assignment]
