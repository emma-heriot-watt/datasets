from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel, Field

from emma_datasets.common import Settings
from emma_datasets.io import read_json


settings = Settings()

TEACH_FRAME_SUFFIX = "jpeg"
TEACH_FRAME_NAME_TEMPLATE = "{agent_name}.frame.{time_start}.{suffix}"


@lru_cache(maxsize=1)
def get_action_idx_to_action_name_map() -> dict[int, str]:
    """Load the mapping from the constants file and cache the results."""
    action_idx_to_action_name_path = settings.paths.constants.joinpath(
        "teach", "action_idx_to_action_name.json"
    )
    action_idx_to_action_name: dict[str, str] = read_json(action_idx_to_action_name_path)

    return {
        int(action_idx): action_name
        for action_idx, action_name in action_idx_to_action_name.items()
    }


@lru_cache(maxsize=1)
def get_agent_id_to_name_map() -> dict[int, str]:
    """Load and process the definitions file and cache the results."""
    default_definitions_path = settings.paths.constants.joinpath(
        "teach", "default_definitions.json"
    )

    definitions_dict = read_json(default_definitions_path)
    agent_definitions_list: list[dict[str, Any]] = definitions_dict["definitions"]["agents"]

    agent_id_to_name_map: dict[int, str] = {
        agent["agent_id"]: agent["agent_name"] for agent in agent_definitions_list
    }

    return agent_id_to_name_map


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
    action_idx: int
    time_start: float = Field(..., ge=0)
    duration: int
    success: Optional[int] = Field(..., ge=0, le=1)
    query: Optional[str] = None
    obj_interaction_action: int
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

    @property
    def agent_name(self) -> str:
        """Get the name of the agent."""
        agent_id_to_name_map = get_agent_id_to_name_map()
        agent_name = agent_id_to_name_map[self.agent_id].lower()
        return agent_name

    @property
    def action_name(self) -> str:
        """Convert the action idx to the action name."""
        action_idx_to_name_map = get_action_idx_to_action_name_map()
        return action_idx_to_name_map[self.action_idx]

    @property
    def frame_path(self) -> str:
        """Convert the interaction into the path to the image frame."""
        return TEACH_FRAME_NAME_TEMPLATE.format(
            agent_name=self.agent_name, time_start=self.time_start, suffix=TEACH_FRAME_SUFFIX
        )

    @property
    def features_path(self) -> str:
        """Convert the interaction into a path to the features file."""
        return TEACH_FRAME_NAME_TEMPLATE.format(
            agent_name=self.agent_name, time_start=self.time_start, suffix="pt"
        )


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
