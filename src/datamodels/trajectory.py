from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel
from pydantic.generics import GenericModel


class Action(BaseModel):
    """Base action model for action trajectories."""

    action: Optional[str]


Low = TypeVar("Low")
High = TypeVar("High")


class GenericActionTrajectory(GenericModel, Generic[Low, High]):
    """Generic Action Trajectory for various datasets."""

    low_level_actions: list[Low]
    high_level_actions: Optional[list[High]]


# TODO(amit):   This needs to be fixed when we know what TEACh's action trajectories look like and
#               there needs to be some common one.
ActionTrajectory = GenericActionTrajectory[Any, Any]
