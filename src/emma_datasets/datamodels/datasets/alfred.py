from typing import Any, Optional

from pydantic import BaseModel, Field, root_validator

from emma_datasets.datamodels.common import Action, Coordinate
from emma_datasets.datamodels.constants import DatasetSplit
from emma_datasets.datamodels.generics import GenericActionTrajectory


class AlfredImageMetadata(BaseModel):
    """Metadata for an image frame from ALFRED Trajectory."""

    high_idx: int
    low_idx: int
    image_name: str


class AlfredInitAction(Coordinate):
    """Metadata of the action used to initialise the agent."""

    action: str
    horizon: int
    rotate_on_teleport: bool = Field(..., alias="rotateOnTeleport")
    rotation: int


class AlfredObjectPosition(Coordinate):
    """Position of object in 3D space."""


class AlfredObjectRotation(Coordinate):
    """The rotation component on an object in 3D space."""


class AlfredObjectPose(BaseModel):
    """Pose of an object in the 3D world."""

    object_name: str = Field(..., alias="objectName")
    position: AlfredObjectPosition
    rotation: AlfredObjectRotation


class AlfredScene(BaseModel):
    """Defines the metadata of an AI2Thor scene used for the current trajectory."""

    dirty_and_empty: bool
    floor_plan: str
    init_action: AlfredInitAction
    object_poses: list[AlfredObjectPose]
    object_toggles: Any
    random_seed: int
    scene_num: int


class AlfredAnnotation(BaseModel):
    """Alfred language annotations associated with each trajectory."""

    high_descs: list[str]
    task_desc: str


class AlfredApiAction(Action):
    """Represents an AI2Thor action that can be executed on the simulartor."""

    object_id: Optional[str] = Field(None, alias="objectId")
    receptacle_object_id: Optional[str] = Field(None, alias="receptacleObjectId")
    force_action: Optional[bool] = Field(None, alias="forceAction")
    move_magnitude: Optional[float] = Field(None, alias="moveMagnitude")


class AlfredPlannerAction(Action):
    """Represents a PDDL planner action."""

    location: Optional[str]
    coordinate_object_id: Optional[tuple[str, list[int]]] = Field(None, alias="coordinateObjectId")
    coordinate_receptable_object_id: Optional[tuple[str, list[int]]] = Field(
        None, alias="coordinateReceptacleObjectId"
    )
    force_visible: Optional[bool] = Field(None, alias="forceVisible")
    object_id: Optional[str] = Field(None, alias="objectId")


class AlfredLowDiscreteAction(Action):
    """Represents a discrete representation of the low-level action used by the planner."""

    args: dict[str, Any]


class AlfredHighDiscreteAction(Action):
    """ALFRED high-level discrete action."""

    args: list[str]


class AlfredHighAction(BaseModel):
    """An ALFRED high-level action definition based on discrete and planner actions."""

    discrete_action: AlfredHighDiscreteAction
    planner_action: AlfredPlannerAction
    high_idx: int


class AlfredLowAction(BaseModel):
    """Low-level AI2Thor action."""

    api_action: AlfredApiAction
    discrete_action: AlfredLowDiscreteAction
    high_idx: int


class AlfredTrajectory(GenericActionTrajectory[AlfredLowAction, AlfredHighAction]):
    """An ALFRED trajectory divided in low-level and high-level actions."""

    low_level_actions: list[AlfredLowAction] = Field(..., alias="low_actions")
    high_level_actions: list[AlfredHighAction] = Field(..., alias="high_pddl")


class AlfredMetadata(BaseModel):
    """Represents the metadata of an ALFRED trajectory.

    For each trajectory, we have multiple language annotations that are stored in
    `turk_annotations`.
    Turkers annotated both goal description as well as sub-goal instructions for each subgoal.

    When using the AI2Thor environment, the scene metadata are required to re-initialise the
    environment in the exact same scenario the trajectory was originally recorded.
    """

    images: list[AlfredImageMetadata]
    plan: AlfredTrajectory
    scene: AlfredScene
    task_id: str
    task_type: str
    turk_annotations: dict[str, list[AlfredAnnotation]]
    dataset_split: Optional[DatasetSplit]

    @root_validator(pre=True)
    @classmethod
    def fix_misaligned_plan(cls, example: dict[str, Any]) -> dict[str, Any]:
        """Removes End subgoal from ALFRED high-level subgoals."""
        high_level_actions = example["plan"]["high_pddl"]
        if high_level_actions[-1]["planner_action"]["action"] == "End":
            example["plan"]["high_pddl"] = high_level_actions[:-1]

        return example
