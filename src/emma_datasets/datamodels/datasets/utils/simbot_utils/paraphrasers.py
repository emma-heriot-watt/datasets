import random

from emma_datasets.datamodels.datasets.utils.simbot_utils.data_augmentations import (
    SimBotObjectAttributes,
)


class InstructionParaphraser:
    """Paraphrase an instruction."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        self.paraphraser_map = {
            "goto": GotoParaphraser(object_synonyms),
            "toggle": ToggleParaphraser(object_synonyms),
            "open": OpenParaphraser(object_synonyms),
            "close": CloseParaphraser(object_synonyms),
            "pickup": PickupParaphraser(object_synonyms),
            "place": PlaceParaphraser(object_synonyms),
        }

    def __call__(
        self, action_type: str, object_id: str, object_attributes: SimBotObjectAttributes
    ) -> str:
        """Paraphrase."""
        paraphraser = self.paraphraser_map.get(action_type, None)
        if paraphraser is None:
            raise AssertionError(f"Action {action_type} cannot be paraphrased")
        insrtuction = paraphraser(object_id, object_attributes)
        return insrtuction


class BaseParaphraser:
    """Base class for a paraphraser."""

    def __init__(self, object_synonyms: dict[str, list[str]], action_type: str) -> None:
        self.object_synonyms = object_synonyms
        self._action_type = action_type
        self._instruction_options: list[str]
        self._available_templates: dict[str, list[str]]
        self._verb_templates = [
            "{verb} the {object}.",
        ]

        self._verb_color_templates = [
            "{verb} the {color} {object}.",
        ]

        self._verb_location_templates = [
            "{verb} the {location} {object}.",
            "{verb} the {object} on your {location}.",
        ]
        self._verb_color_location_templates = [
            "{verb} the {color} {location} {object}.",
            "{verb} the {location} {color} {object}.",
            "{verb} the {object} on your {location}.",
        ]

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Paraphrase."""
        raise NotImplementedError

    def _get_instruction(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        available_types: list[str],
    ) -> str:
        selected_type = random.choice(available_types)
        selected_template = random.choice(self._available_templates[selected_type])

        if selected_type == self._action_type:
            object_name = attributes.readable_name
        else:
            object_name = random.choice(self.object_synonyms[object_id])

        template_values = {
            "verb": random.choice(self._instruction_options),
            "object": object_name,
            "color": attributes.color,
            "location": attributes.location,
        }
        instruction = selected_template.format(**template_values)

        return instruction.lower()


class GotoParaphraser(BaseParaphraser):
    """This is called in training only!"""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="goto")

        self._instruction_options = [
            "go to",
            "go towards",
            "move to",
            "move towards",
            "head to",
            "head towards",
            "approach",
        ]

        self._available_templates = {
            "goto": self._verb_templates,
            "goto_color": self._verb_color_templates,
            "goto_location": self._verb_location_templates,
            "goto_color_location": self._verb_color_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a goto instruction."""
        available_types = ["goto"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("goto_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("goto_location")

        if object_color is not None and object_location is not None:
            available_types.append("goto_color_location")
        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction


class ToggleParaphraser(BaseParaphraser):
    """Paraphrase toggle instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="toggle")
        self._instruction_options = [
            "toggle",
            "start",
            "activate",
            "turn on",
            "switch on",
            "turn off",
            "switch off",
            "power up",
        ]

        self._available_templates = {
            "toggle": self._verb_templates,
            "toggle_color": self._verb_color_templates,
            "toggle_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a toggle instruction."""
        available_types = ["toggle"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("toggle_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("toggle_location")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction


class OpenParaphraser(BaseParaphraser):
    """Paraphrase open instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="open")
        self._instruction_options = ["open"]

        self._available_templates = {
            "open": self._verb_templates,
            "open_color": self._verb_color_templates,
            "open_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a open instruction."""
        available_types = ["open"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("open_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("open_location")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction


class CloseParaphraser(BaseParaphraser):
    """Paraphrase close instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="close")
        self._instruction_options = ["close"]

        self._available_templates = {
            "close": self._verb_templates,
            "close_color": self._verb_color_templates,
            "close_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a close instruction."""
        available_types = ["close", "shut"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("close_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("close_location")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction


class PickupParaphraser(BaseParaphraser):
    """Paraphrase pickup instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="pickup")
        self._instruction_options = [
            "pick up",
            "take",
            "grab",
            "collect",
            "get",
        ]

        self._available_templates = {
            "pickup": self._verb_templates,
            "pickup_color": self._verb_color_templates,
            "pickup_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a pickup instruction."""
        available_types = ["pickup"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("pickup_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("pickup_location")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction


class PlaceParaphraser(BaseParaphraser):
    """Paraphrase place instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="place")
        self._instruction_options = [
            "place it on",
            "leave it on",
            "put it on",
            "set it on",
            "put it down on",
        ]

        self._available_templates = {
            "place": self._verb_templates,
            "place_color": self._verb_color_templates,
            "place_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a place instruction."""
        available_types = ["place"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("place_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("place_location")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction
