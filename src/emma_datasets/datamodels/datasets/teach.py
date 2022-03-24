from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, cast

from pydantic import BaseModel, Field, PrivateAttr

from emma_datasets.common import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import MediaType
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

    @property
    def object_name(self) -> Optional[str]:
        """Get the object name that the agent is interacting with.

        The `oid` is in the form `CoffeeMachine|-02.94|+00.93|+03.61`, and we only want the first
        part of it.
        """
        if self.oid is None:
            return None

        return self.oid.split("|")[0]


class ExtendedTeachDriverAction(TeachDriverAction):
    """Extended version of the driver action with utterance."""

    utterance: Optional[str] = None


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

    @property
    def object_name(self) -> Optional[str]:
        """Get the object name that the agent is interacting with.

        The `oid` is in the form `CoffeeMachine|-02.94|+00.93|+03.61`, and we only want the first
        part of it.
        """
        if self.oid is None:
            return None

        return self.oid.split("|")[0]


class TeachUtterance(BaseModel):
    """Model for an utterance from the dialogue history.

    This is used as an easy way to validate the inputs for the dialogue history attributes.
    """

    __root__: list[str] = Field(..., min_items=2, max_items=2)

    @property
    def speaker(self) -> Literal["Driver", "Commander"]:
        """Get the speaker for the utterance."""
        speaker = self.__root__[0]

        if speaker in {"Driver", "Commander"}:
            return cast(Literal["Driver", "Commander"], speaker)

        raise ValueError("Value for speaker is not either 'Driver' or 'Commander'.")

    @property
    def utterance(self) -> str:
        """Get the utterance itself."""
        return self.__root__[1]


class TeachEdhInstance(BaseInstance):
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

    # Subgoals
    history_subgoals: list[str]
    future_subgoals: list[str]

    # State
    expected_init_goal_conditions_satisfied: int
    expected_init_goal_conditions_total: int

    init_state_diff: Any
    final_state_diff: Any
    state_changes: Any

    _features_path: Path = PrivateAttr()
    _future_features_path: Path = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        self._features_path = settings.paths.teach_edh_features.joinpath(  # noqa: WPS601
            f"{self.instance_id}.history.pt"
        )
        self._future_features_path = settings.paths.teach_edh_features.joinpath(  # noqa: WPS601
            f"{self.instance_id}.future.pt"
        )

    @property
    def modality(self) -> MediaType:
        """Get the modality of the instance."""
        return MediaType.video

    @property
    def features_path(self) -> Path:
        """Get the path to the features for this instance."""
        return self._features_path

    @property
    def future_features_path(self) -> Path:
        """Get the path to the features from future driver actions for this instance."""
        return self._future_features_path

    @property
    def extended_driver_action_history(self) -> list[ExtendedTeachDriverAction]:
        """Get extended driver action history using the cleaned dialog history.

        We need to have a counter of every `Text` action that has happened to be sure to get the
        correct utterance for the action.
        """
        action_history: list[ExtendedTeachDriverAction] = []
        utterance_counter = 0

        for action in self.driver_action_history:
            action_dict = action.dict()

            if action.action_id == 100:
                action_dict["utterance"] = self._driver_dialog_history[utterance_counter]
                utterance_counter += 1

            action_history.append(ExtendedTeachDriverAction(**action_dict))

        return action_history

    @property
    def interaction_history(self) -> list[TeachInteraction]:
        """Get all interactions that happened in the past."""
        return [
            interaction
            for interaction in self.interactions
            if not self._is_interaction_in_future(interaction)
        ]

    @property
    def interactions_future(self) -> list[TeachInteraction]:
        """Get all interactions which happened 'in the future'."""
        return [
            interaction
            for interaction in self.interactions
            if not self._is_interaction_in_future(interaction)
        ]

    def _is_interaction_in_future(self, interaction: TeachInteraction) -> bool:
        """Returns True if the given interaction is 'in the future'."""
        return interaction.time_start > self._last_action_time

    @property
    def _last_action_time(self) -> float:
        """Get the last time, after which all interactions will be be in the future."""
        return self.driver_action_history[-1].time_start

    @property
    def _driver_dialog_history(self) -> list[str]:
        """Get the dialog history of only the driver."""
        return [
            utterance.utterance
            for utterance in self.dialog_history
            if utterance.speaker == "Driver"
        ]
