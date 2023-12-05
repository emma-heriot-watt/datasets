from typing import Generic, Optional, TypeVar

from pydantic.generics import GenericModel


Low = TypeVar("Low")
High = TypeVar("High")


class GenericActionTrajectory(GenericModel, Generic[Low, High]):
    """Generic Action Trajectory for various datasets."""

    low_level_actions: list[Low]
    high_level_actions: Optional[list[High]]
