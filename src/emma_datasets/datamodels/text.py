from typing import Union

from emma_datasets.datamodels.base_model import BaseModel


class Caption(BaseModel):
    """Text caption for the image."""

    text: str


class QuestionAnswerPair(BaseModel):
    """Question-Answer pair for image."""

    id: str
    question: str
    answer: str


Text = Union[Caption, QuestionAnswerPair]
