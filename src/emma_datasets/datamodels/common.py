from emma_datasets.datamodels.base_model import BaseModel


class Coordinate(BaseModel):
    """Model for coordinates."""

    x: float
    y: float
    z: float


class Action(BaseModel):
    """Base action model for action trajectories."""

    action: str
