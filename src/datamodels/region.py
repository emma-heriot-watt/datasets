from typing import Union

import numpy
from numpy.typing import NDArray
from pydantic import validator

from src.datamodels.base_model import BaseModel


BBox = NDArray[numpy.float32]


class Region(BaseModel):
    """Regions within media sources, with additional information."""

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
