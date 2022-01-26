from typing import Optional

from emma_datasets.datamodels.base_model import BaseModel
from emma_datasets.datamodels.datasets import GqaObject


class SceneGraph(BaseModel):
    """Scene graph for a scene.

    Currently, this is just a reduced version of the `GQASceneGraph`. Does the scene graph
    representation need improving?
    """

    location: Optional[str]
    weather: Optional[str]
    objects: dict[str, GqaObject]  # noqa: WPS110
