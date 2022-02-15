from typing import Generic, Optional, TypeVar

from pydantic import BaseModel
from pydantic.generics import GenericModel


class Coordinate(BaseModel):
    """Model for coordinates."""

    x: float
    y: float
    z: float


class Action(BaseModel):
    """Base action model for action trajectories."""

    action: str


Low = TypeVar("Low")
High = TypeVar("High")


class GenericActionTrajectory(GenericModel, Generic[Low, High]):
    """Generic Action Trajectory for various datasets."""

    low_level_actions: list[Low]
    high_level_actions: Optional[list[High]]
