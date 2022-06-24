from pathlib import Path
from typing import Optional

from emma_datasets.datamodels.datasets.vqa_v2 import VQAv2Instance, load_vqa_v2_annotations


def test_can_load_vqa_v2_data(vqa_v2_all_data_paths: list[tuple[Path, Optional[Path]]]) -> None:
    for paths in vqa_v2_all_data_paths:
        question_path = paths[0]
        answers_path = paths[1]
        assert question_path.exists()
        if answers_path is not None:
            assert answers_path.exists()

        raw_instances = load_vqa_v2_annotations(question_path, answers_path)
        assert len(raw_instances)


def test_vqa_v2_data_has_custom_attributes(
    vqa_v2_all_data_paths: list[tuple[Path, Optional[Path]]]
) -> None:
    for paths in vqa_v2_all_data_paths:
        question_path = paths[0]
        answers_path = paths[1]
        raw_instances = load_vqa_v2_annotations(question_path, answers_path)

        for raw_instance in raw_instances:
            parsed_instance = VQAv2Instance.parse_obj(raw_instance)

            assert parsed_instance
            assert isinstance(parsed_instance.image_id, str)
            assert len(parsed_instance.image_id)
            assert isinstance(parsed_instance.question_id, str)
            assert len(parsed_instance.question_id)
            assert isinstance(parsed_instance.question, str)
            assert len(parsed_instance.question)
            assert isinstance(parsed_instance.features_path, Path)

            if parsed_instance.answers is None:
                assert (
                    answers_path is None
                ), "Answers should be None only if the annotation file was not provided"
                assert parsed_instance.answer_type is None
            else:
                assert (
                    len(parsed_instance.answers) == 10
                ), "VQA-v2 instances should have 10 answers"
                assert all([isinstance(answer, str) for answer in parsed_instance.answers])
                assert isinstance(parsed_instance.answer_type, str)
                assert len(parsed_instance.answer_type)
