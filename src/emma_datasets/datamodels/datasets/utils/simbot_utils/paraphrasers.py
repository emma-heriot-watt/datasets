import random

from emma_datasets.constants.simbot.simbot import get_arena_definitions, get_objects_asset_synonyms
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_asset_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    ParaphrasableActions,
    SimBotInstructionInstance,
    SimBotObjectAttributes,
)


class InstructionParaphraser:
    """Paraphrase an instruction."""

    def __init__(self) -> None:
        object_synonyms = get_objects_asset_synonyms()
        self.paraphraser_map = {
            "goto": GotoParaphraser(object_synonyms),
            "toggle": ToggleParaphraser(object_synonyms),
            "open": OpenParaphraser(object_synonyms),
            "close": CloseParaphraser(object_synonyms),
            "pickup": PickupParaphraser(object_synonyms),
            "place": PlaceParaphraser(object_synonyms),
            "break": BreakParaphraser(object_synonyms),
            "scan": ScanParaphraser(object_synonyms),
            "pour": PourParaphraser(object_synonyms),
            "clean": CleanParaphraser(object_synonyms),
            "fill": FillParaphraser(object_synonyms),
            "search": SearchParaphraser(object_synonyms),
        }

    def __call__(
        self,
        action_type: str,
        object_id: str,
        object_attributes: SimBotObjectAttributes,
    ) -> str:
        """Paraphrase."""
        paraphraser = self.paraphraser_map.get(action_type, None)
        if paraphraser is None:
            raise AssertionError(f"Action {action_type} cannot be paraphrased")
        instruction = paraphraser(object_id, object_attributes)
        return instruction

    def from_instruction_instance(self, instruction_instance: SimBotInstructionInstance) -> str:
        """Paraphrase an instruction from a SimbotInstructionInstance."""
        cond1 = len(instruction_instance.actions) == 1
        action = instruction_instance.actions[0]
        action_type = action.type.lower()
        action_data = action.get_action_data
        cond2 = action_type in ParaphrasableActions
        if cond1 and cond2:
            # For instruction instances that have multiple objects e.g, search we pick one at random
            if isinstance(action_data["object"]["attributes"], list):
                object_candidates = len(action_data["object"]["attributes"])
                object_candidate_index = random.randint(0, object_candidates - 1)
                object_attributes = SimBotObjectAttributes(
                    **action_data["object"]["attributes"][object_candidate_index]
                )
                object_id = action_data["object"]["id"][object_candidate_index]
            else:
                object_attributes = SimBotObjectAttributes(**action_data["object"]["attributes"])
                object_id = action_data["object"]["id"]

            instruction = self(
                action_type=action_type,
                object_id=object_id,
                object_attributes=object_attributes,
            )
        else:
            instruction = instruction_instance.instruction.instruction
        return instruction


class BaseParaphraser:
    """Base class for a paraphraser."""

    def __init__(self, object_synonyms: dict[str, list[str]], action_type: str) -> None:
        self.object_synonyms = object_synonyms
        self._action_type = action_type
        self._instruction_options: list[str]
        self._available_templates: dict[str, list[str]]
        self._assets_to_labels = get_arena_definitions()["asset_to_label"]
        self._special_name_cases = {
            "V_Monitor_Embiggenator",
            "V_Monitor_Gravity",
            "V_Monitor_Laser",
            "V_Monitor_FreezeRay",
        }
        self._full_templates = [
            "{instruction}",
        ]
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

        if selected_type == self._action_type and selected_type in self._special_name_cases:
            object_name = attributes.readable_name
        else:
            object_asset = get_object_asset_from_object_id(object_id, self._assets_to_labels)
            object_name = random.choice(self.object_synonyms.get(object_asset, [object_asset]))

        template_values = {
            "verb": random.choice(self._instruction_options),
            "object": object_name,
            "color": attributes.color,
            "location": attributes.location,
        }
        instruction = selected_template.format(**template_values)

        return instruction.lower()

    def _add_prefix(self, instruction: str, prefix: str) -> str:
        return f"{prefix} {instruction}"

    def _add_suffix(self, instruction: str, suffix: str) -> str:
        if instruction.endswith("."):
            instruction = instruction[:-1]
        return f"{instruction} {suffix}"


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
        self._instruction_options = ["close", "shut"]

        self._available_templates = {
            "close": self._verb_templates,
            "close_color": self._verb_color_templates,
            "close_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a close instruction."""
        available_types = ["close"]
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
        # TODO: Use random nouns to place, e.g. Place the donut on the ....
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


class BreakParaphraser(BaseParaphraser):
    """Paraphrase break instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="break")
        self._instruction_options = [
            "break",
            "smash",
            "shatter",
            "crash",
        ]
        self._prefix_option = "use the hammer to"
        self._suffix_option = "with the hammer."

        self._available_templates = {
            "break": self._verb_templates,
            "break_color": self._verb_color_templates,
            "break_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a break instruction."""
        available_types = ["break"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("break_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("break_location")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        proba = random.random()
        if proba < (1 / 3):
            instruction = self._add_prefix(instruction, self._prefix_option)
        elif proba < (2 / 3):
            instruction = self._add_suffix(instruction, self._suffix_option)
        return instruction


class CleanParaphraser(BaseParaphraser):
    """Paraphrase clean instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="clean")
        self._instruction_options = [
            "clean the plate.",
            "rinse the plate.",
        ]
        self._suffix_option = "in the sink."

        self._available_templates = {
            "clean": self._full_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a clean instruction."""
        available_types = ["clean"]

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        if random.random() < (1 / 2):
            instruction = self._add_suffix(instruction, self._suffix_option)
        return instruction


class PourParaphraser(BaseParaphraser):
    """Paraphrase pour instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="pour")
        self._instruction_options = [
            "pour it into",
            "pour it in",
            "pour the water into",
            "pour the coffee into",
            "pour the cereal into",
            "pour the milk into",
        ]

        self._available_templates = {
            "pour": self._verb_templates,
            "pour_color": self._verb_color_templates,
            "pour_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a pour instruction."""
        available_types = ["pour"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("pour_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("pour_location")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction


class ScanParaphraser(BaseParaphraser):
    """Paraphrase scan instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="scan")
        self._instruction_options = ["Scan"]

        self._available_templates = {
            "scan": self._verb_templates,
            "scan_color": self._verb_color_templates,
            "scan_location": self._verb_location_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a scan instruction."""
        available_types = ["scan"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("scan_color")

        object_location = attributes.location
        if object_location is not None:
            available_types.append("scan_location")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction


class FillParaphraser(BaseParaphraser):
    """Paraphrase fill instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="fill")
        self._instruction_options = ["fill"]

        self._suffix_option = "with water."
        self._available_templates = {"fill": self._verb_templates}

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a fill instruction."""
        available_types = ["fill"]

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction


class SearchParaphraser(BaseParaphraser):
    """Paraphrase search instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="search")
        self._instruction_options = [
            "find",
            "locate",
            "where is",
            "search for",
            "seek",
            "trace",
            "can you find",
            "can you locate",
            "can you search for",
            "do you see",
        ]

        self._available_templates = {
            "search": self._verb_templates,
            "search_color": self._verb_color_templates,
        }

    def __call__(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        """Get a search instruction."""
        available_types = ["search"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("search_color")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction
