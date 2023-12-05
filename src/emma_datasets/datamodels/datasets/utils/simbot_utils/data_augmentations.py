import random
from typing import Any, Optional

from emma_datasets.common.settings import Settings
from emma_datasets.constants.simbot.simbot import get_low_level_action_templates


settings = Settings()


class SyntheticLowLevelActionSampler:
    """Create synthetic examples of low level actions."""

    def __init__(self) -> None:
        self._low_level_action_templates = get_low_level_action_templates()
        self._low_level_actions = list(self._low_level_action_templates.keys())

    def __call__(
        self,
        mission_id: str,
        annotation_id: str,
        instruction_idx: int,
        original_action: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Sample a low level action and an instruction template."""
        if original_action is None:
            raise AssertionError("Need the original actions")
        low_level_action = random.choice(self._low_level_actions)
        low_level_action_template = random.choice(
            self._low_level_action_templates[low_level_action]["templates"]
        )
        action_type = self._low_level_action_templates[low_level_action]["type"]
        action_id = original_action["id"]
        color_images = original_action["colorImages"]
        payload = {"direction": self._low_level_action_templates[low_level_action]["direction"]}

        synthetic_instruction = {
            "instruction": low_level_action_template,
            "actions": [action_id],
        }
        synthetic_action = {
            "id": action_id,
            "type": action_type,
            action_type.lower(): payload,
            "colorImages": color_images,
            "final": True,
        }
        instruction_dict = {
            "instruction": synthetic_instruction,
            "actions": [synthetic_action],
            "mission_id": mission_id,
            "annotation_id": annotation_id,
            "instruction_id": str(instruction_idx),
            "synthetic": True,
        }
        return instruction_dict
