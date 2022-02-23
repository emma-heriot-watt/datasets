from typing import Any, Optional

from pydantic import BaseModel, Field


class TeachDriverAction(BaseModel):
    """Driver Action for TEACh."""

    action_id: int
    action_idx: int
    obj_interaction_action: int
    action_name: str
    time_start: float
    oid: Optional[str] = None
    x: Optional[float]
    y: Optional[float]


class TeachInteraction(BaseModel):
    """Model for a single interaction within TEACh."""

    agent_id: int = Field(..., ge=0, le=1)
    action_id: int
    time_start: float = Field(..., ge=0)
    duration: int
    success: Optional[int] = Field(..., ge=0, le=1)
    query: Optional[str] = None
    obj_interaction_action: int
    action_idx: int
    utterance: Optional[str] = None
    corrected_utterance: Optional[str] = None
    is_corrected: Optional[int] = None
    pose_delta: Optional[list[float]] = None
    pose: Optional[list[float]] = None
    x: Optional[float] = None
    y: Optional[float] = None
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    oid: Optional[str] = None


class TeachUtterance(BaseModel):
    """Model for an utterance from the dialogue history.

    This is used as an easy way to validate the inputs for the dialogue history attributes.
    """

    __root__: list[str] = Field(..., min_items=2, max_items=2)


class TeachEdhInstance(BaseModel):
    """TEACh EDH Instance."""

    game_id: str
    instance_id: str

    pred_start_idx: int

    # Dialogue
    dialog_history: list[TeachUtterance]
    dialog_history_cleaned: list[TeachUtterance]

    # Images
    driver_image_history: list[str]
    driver_images_future: list[str]

    # Interactions
    interactions: list[TeachInteraction]

    # Actions
    driver_action_history: list[TeachDriverAction]
    driver_actions_future: list[TeachDriverAction]

    expected_init_goal_conditions_total: int
    expected_init_goal_conditions_satisfied: int

    # Subgoals
    history_subgoals: list[str]
    future_subgoals: list[str]

    # State
    init_state_diff: Any
    final_state_diff: Any
    state_changes: Any
