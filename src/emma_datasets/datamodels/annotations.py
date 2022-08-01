import re
from typing import Optional, Union

import numpy
from numpy.typing import NDArray
from pydantic import validator

from emma_datasets.datamodels.base_model import BaseModel
from emma_datasets.datamodels.constants import AnnotationType
from emma_datasets.datamodels.datasets.alfred import AlfredHighAction, AlfredLowAction
from emma_datasets.datamodels.datasets.gqa import GqaObject
from emma_datasets.datamodels.generics import GenericActionTrajectory


BBox = NDArray[numpy.float32]


class Annotation(BaseModel):
    """Base annotation used by other annotation interfaces."""

    _annotation_type: AnnotationType

    def get_language_data(self) -> Union[str, list[str]]:
        """Get the language data from the current Annotation class."""
        raise NotImplementedError()


class Caption(Annotation):
    """Text caption for the image."""

    _annotation_type = AnnotationType.caption

    text: str

    def get_language_data(self) -> str:
        """Get the language data from a Caption."""
        return self.text


class QuestionAnswerPair(Annotation):
    """Question-Answer pair for image."""

    _annotation_type = AnnotationType.qa_pair

    id: str
    question: str
    answer: Union[str, list[str]]

    def get_language_data(self) -> str:
        """Get the language data from a QA Pair."""
        return f"{self.question} {self.answer}"


Text = Union[Caption, QuestionAnswerPair]


class SceneGraph(Annotation):
    """Scene graph for a scene.

    Currently, this is just a reduced version of the `GQASceneGraph`. Does the scene graph
    representation need improving?
    """

    _annotation_type = AnnotationType.scene_graph

    location: Optional[str]
    weather: Optional[str]
    objects: dict[str, GqaObject]  # noqa: WPS110

    def get_language_data(self) -> list[str]:
        """Get the language data from a Scene Graph."""
        annotations = []

        for scene_obj in self.objects.values():
            if scene_obj.attributes:
                for attr in scene_obj.attributes:
                    annotations.append(f"{scene_obj.name} has attribute {attr}")

            if scene_obj.relations:
                for rel in scene_obj.relations:
                    rel_object = self.objects[rel.object]
                    annotations.append(f"{scene_obj.name} {rel.name} {rel_object.name}")

        return annotations


class Region(Annotation):
    """Regions within media sources, with additional information."""

    _annotation_type = AnnotationType.region

    # Is 1D with 4 values (x, y, width, height) with x,y being top-left coordinate
    bbox: BBox
    caption: str

    @validator("bbox", pre=True)
    @classmethod
    def convert_bbox_to_numpy(cls, bbox: Union[list[int], list[float], BBox]) -> BBox:
        """Convert list of numbers to a numpy array before validation.

        If stored in a file, it is likely as a list of numbers, so they are then converted back to
        a numpy array. If it's not a list, it'll just return whatever it is.
        """
        if isinstance(bbox, list):
            return numpy.asarray(bbox, dtype=numpy.float32)
        return bbox

    @validator("bbox")
    @classmethod
    def bbox_has_positive_numbers(cls, bbox: BBox) -> BBox:
        """Verify bbox only has 4 positive numbers.

        This is not true for VG, so this has been disabled for now.
        """
        # if not numpy.all(numpy.greater_equal(bbox, 0)):
        #     raise AssertionError("All numbers within a BBox should be greater than 0.")
        return bbox

    @property
    def x_coord(self) -> int:
        """Get the top-left x coordinate of the region."""
        return self.bbox[0]

    @property
    def y_coord(self) -> int:
        """Get the top-left y coordinate of the region."""
        return self.bbox[1]

    @property
    def width(self) -> int:
        """Get the width of the region."""
        return self.bbox[2]

    @property
    def height(self) -> int:
        """Get the height of the region."""
        return self.bbox[3]

    def get_language_data(self) -> str:
        """Get the language data from a Region."""
        return self.caption


class ActionTrajectory(GenericActionTrajectory[AlfredLowAction, AlfredHighAction]):
    """Action Trajectory used for the standardised annotation."""

    _annotation_type = AnnotationType.action_trajectory

    low_level_actions: list[AlfredLowAction]
    high_level_actions: list[AlfredHighAction]

    def get_language_data(self) -> str:
        """Get the language data from an action trajectory."""
        trajectory_str = " ".join(
            self._get_action_string(low_action.discrete_action.action)
            for low_action in self.low_level_actions
        )

        return trajectory_str

    def _get_action_string(self, action_name: str) -> str:
        """Returns a phrase associated with the action API name.

        API action names are in camelcase format: MoveAhead_25
        """
        parts: list[str] = []

        for x in re.findall("[A-Z][^A-Z]*", action_name):
            parts.extend(xi for xi in x.split("_"))

        return " ".join(parts)


class TaskDescription(Annotation):
    """Text caption for the image."""

    _annotation_type = AnnotationType.task_description

    text: str

    def get_language_data(self) -> str:
        """Get the language data from a TaskDescription."""
        return self.text
