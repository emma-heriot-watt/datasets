from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, PrivateAttr

from emma_datasets.common import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import DatasetName, DatasetSplit, MediaType
from emma_datasets.datamodels.datasets.utils.vqa_v2_utils import normalize_answer
from emma_datasets.io import read_json


settings = Settings()

VQAv2AnnotationsType = list[dict[str, Any]]


class VQAv2AnnotationPaths(BaseModel):
    """VQA-v2 annotation paths for a dataset split."""

    split: DatasetSplit
    questions_path: Path
    answers_path: Optional[Path]


def get_vqa_v2_annotation_paths(vqa_v2_instances_base_dir: Path) -> list[VQAv2AnnotationPaths]:
    """Get annotation paths for all VQA-v2 splits."""
    vqa_v2_dir_paths = [
        VQAv2AnnotationPaths(
            split=DatasetSplit.train,
            questions_path=vqa_v2_instances_base_dir.joinpath(
                "v2_OpenEnded_mscoco_train2014_questions.json"
            ),
            answers_path=vqa_v2_instances_base_dir.joinpath(
                "v2_mscoco_train2014_annotations.json"
            ),
        ),
        VQAv2AnnotationPaths(
            split=DatasetSplit.valid,
            questions_path=vqa_v2_instances_base_dir.joinpath(
                "v2_OpenEnded_mscoco_val2014_questions.json"
            ),
            answers_path=vqa_v2_instances_base_dir.joinpath("v2_mscoco_val2014_annotations.json"),
        ),
        VQAv2AnnotationPaths(
            split=DatasetSplit.test_dev,
            questions_path=vqa_v2_instances_base_dir.joinpath(
                "v2_OpenEnded_mscoco_test-dev2015_questions.json"
            ),
            answers_path=None,
        ),
        VQAv2AnnotationPaths(
            split=DatasetSplit.test,
            questions_path=vqa_v2_instances_base_dir.joinpath(
                "v2_OpenEnded_mscoco_test2015_questions.json"
            ),
            answers_path=None,
        ),
    ]
    return vqa_v2_dir_paths


def read_vqa_v2_json(
    annotation_path: Path, annotation_type: Literal["questions", "annotations"]
) -> dict[str, Any]:
    """Load the VQA-v2 annotations as a dictionary with question ids as keys."""
    annotation_list = read_json(annotation_path)[annotation_type]
    annotations = {str(instance["question_id"]): instance for instance in annotation_list}
    return annotations


class VQAv2Target(BaseModel):
    """VQA-v2 answers including the answer id and score."""

    answer: str
    target_id: Optional[int]
    score: Optional[float]


def vqa_v2_score(count: int) -> float:
    """VQA-v2 includes 10 answers for each question.

    Scores are assigned as follows:
    - 0.3 if the answer appears once
    - 0.6 if the answer appears twice
    - 0.9 if the answer appears three times
    - 1.0 if the answer appears more than three times
    """
    return min(1.0, round(0.3 * count, 1))  # noqa: WPS432


def prepare_training_targets(answers: list[str], ans2label: dict[str, int]) -> list[VQAv2Target]:
    """Compute answer VQA scores for answers in the predifined candidates."""
    answers = [normalize_answer(answer) for answer in answers]
    targets = []
    for answer, count in Counter(answers).items():
        label = ans2label.get(answer, -1)
        if label > 0:
            targets.append(
                VQAv2Target(
                    target_id=label,
                    score=vqa_v2_score(count),
                    answer=answer,
                )
            )
    return targets


def merge_vqa_v2_annotations(
    questions: dict[str, Any], all_answers: dict[str, Any]
) -> dict[str, Any]:
    """Merge question and answer annotations for VQA-v2."""
    ans2label_path = settings.paths.constants.joinpath("vqa_v2_ans2label.json")
    ans2label = read_json(ans2label_path)
    for question_id in questions.keys():
        answers = all_answers.get(question_id, None)
        if answers is None:
            raise AssertionError(f"Annotations for question {question_id} not found!")
        questions[question_id]["answer_type"] = answers["answer_type"]
        questions[question_id]["question_type"] = answers["question_type"]
        # Keep only the answers, discard the answer condfindence and id
        questions[question_id]["answers"] = [answer["answer"] for answer in answers["answers"]]
        # All VQA-v2 instances should have 10 answers
        if len(questions[question_id]["answers"]) != 10:
            raise AssertionError(
                f"Found {len(questions[question_id]['answers'])} answers instead of 10!"
            )

        questions[question_id]["training_targets"] = prepare_training_targets(
            questions[question_id]["answers"], ans2label
        )

    return questions


def load_vqa_v2_annotations(
    questions_path: Path, answers_path: Optional[Path]
) -> VQAv2AnnotationsType:
    """Load question and answer annotations for VQA-v2.

    Question and answer annotations are saved in separate files, but they can be merged based on
    their unique question id.
    """
    questions = read_vqa_v2_json(questions_path, "questions")
    for question_id in questions.keys():
        questions[question_id]["original_dataset"] = DatasetName.vqa_v2

    if answers_path is not None:
        answers = read_vqa_v2_json(answers_path, "annotations")
        questions = merge_vqa_v2_annotations(questions=questions, all_answers=answers)

    return list(questions.values())


def resplit_vqa_v2_annotations(
    train_annotations: VQAv2AnnotationsType, valid_annotations: VQAv2AnnotationsType
) -> tuple[VQAv2AnnotationsType, VQAv2AnnotationsType]:
    """Resplit train and valiadtion data to use more data for training."""
    pass  # noqa: WPS420


def load_vqa_visual_genome_annotations() -> VQAv2AnnotationsType:
    """Load additional visual genome data."""
    pass  # noqa: WPS420


class VQAv2Instance(BaseInstance):
    """VQA-v2 Instance."""

    image_id: str
    question_id: str
    question: str
    question_type: Optional[str]
    answers: Optional[list[str]]
    answer_type: Optional[str]
    training_targets: Optional[list[VQAv2Target]]
    original_dataset: DatasetName
    _features_path: Path = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.original_dataset == DatasetName.vqa_v2:
            self._features_path = settings.paths.coco_features.joinpath(  # noqa: WPS601
                f"{self.image_id.zfill(12)}.pt"  # noqa: WPS432
            )

    @property
    def modality(self) -> MediaType:
        """Get the modality of the instance."""
        return MediaType.image

    @property
    def features_path(self) -> Path:
        """Get the path to the features for this instance."""
        return self._features_path
