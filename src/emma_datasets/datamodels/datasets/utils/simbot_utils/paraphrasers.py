import random
from copy import deepcopy
from typing import Optional

from emma_datasets.constants.simbot.simbot import (
    get_arena_definitions,
    get_objects_asset_synonyms,
    get_pickable_objects_ids,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.instruction_processing import (
    get_object_asset_from_object_id,
    get_object_readable_name_from_object_id,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    ParaphrasableActions,
    SimBotInstructionInstance,
    SimBotObjectAttributes,
)


class InventoryObjectGenerator:
    """Generate an object that could be in the agent inventory for each instruction."""

    def __init__(self) -> None:
        pickable_objects = get_pickable_objects_ids()
        # Note that pickup is missing in purpose from inventory_choices
        self.inventory_choices = {
            "goto": pickable_objects,
            "toggle": pickable_objects,
            "open": pickable_objects,
            "close": pickable_objects,
            "place": pickable_objects,
            "scan": pickable_objects,
            "break": ["Hammer"],
            "pour": [
                "CoffeeMug_Yellow",
                "CoffeeMug_Boss",
                "CoffeePot_01",
                "Bowl_01",
                "MilkCarton_01",
                "CoffeeBeans_01",
            ],
            "clean": ["FoodPlate_01"],
            "fill": ["CoffeeMug_Yellow", "CoffeeMug_Boss", "CoffeePot_01", "Bowl_01"],
            "search": pickable_objects,
        }

    def __call__(self, action_type: str) -> Optional[str]:
        """Get a random object."""
        action_inventory_choices = self.inventory_choices.get(action_type.lower(), None)
        if action_inventory_choices is None or not action_inventory_choices:
            return None
        return random.choice(action_inventory_choices)


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
        self._inventory_object_generator = InventoryObjectGenerator()

    def __call__(
        self,
        action_type: str,
        object_id: str,
        object_attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
        """Paraphrase."""
        paraphraser = self.paraphraser_map.get(action_type, None)
        if paraphraser is None:
            raise AssertionError(f"Action {action_type} cannot be paraphrased")
        if paraphraser.requires_inventory and inventory_object_id is None:
            inventory_object_id = self._inventory_object_generator(action_type=action_type)
        instruction = paraphraser(object_id, object_attributes, inventory_object_id)
        return instruction

    def from_instruction_instance(
        self, instruction_instance: SimBotInstructionInstance
    ) -> tuple[str, Optional[str]]:
        """Paraphrase an instruction from a SimbotInstructionInstance."""
        cond1 = len(instruction_instance.actions) == 1
        action = instruction_instance.actions[0]
        action_type = action.type.lower()
        action_data = action.get_action_data
        cond2 = action_type in ParaphrasableActions
        inventory_object_id = None
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

            inventory_object_id = self.sample_inventory_object(action_type=action_type)
            paraphraser = self.paraphraser_map.get(action_type, None)
            if paraphraser is None:
                raise AssertionError(f"Action {action_type} cannot be paraphrased")
            instruction = paraphraser(object_id, object_attributes, inventory_object_id)
        else:
            instruction = instruction_instance.instruction.instruction
        return instruction, inventory_object_id

    def sample_inventory_object(self, action_type: str) -> Optional[str]:
        """Sample an inventory object."""
        paraphraser = self.paraphraser_map.get(action_type, None)
        if paraphraser is None:
            return None
        # If the action type does not require an inventory object, set it with probability 0.5
        if paraphraser.requires_inventory or random.random() < 1 / 2:
            return self._inventory_object_generator(action_type=action_type)
        return None

    def is_inventory_required(self, action_type: str) -> bool:
        """Is the inventory required for the action?"""
        paraphraser = self.paraphraser_map.get(action_type, None)
        if paraphraser is None:
            return False
        return paraphraser.requires_inventory


class BaseParaphraser:
    """Base class for a paraphraser."""

    def __init__(self, object_synonyms: dict[str, list[str]], action_type: str) -> None:
        self.object_synonyms = object_synonyms
        self._action_type = action_type
        self._instruction_options: list[str]
        # Additional instruction options that cannot be combined with a prefix
        self._no_prefix_instruction_options: list[str] = []
        self._available_templates: dict[str, list[str]]
        arena_definitions = get_arena_definitions()
        self._assets_to_labels = arena_definitions["asset_to_label"]
        self._special_name_cases = arena_definitions["special_asset_to_readable_name"]
        self._full_templates = [
            # By convention the full instruction will be provided in `verb` entry.
            "{verb}",
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
        self._prefix_options = [
            "I would like to",
            "I need to",
            "I need you to",
            "I am telling you to",
            "you should",
            "we need to",
            "let's",
            "can you",
            "could you",
            "okay",
            "okay now",
            "now",
            "please",
        ]
        self.requires_inventory = False

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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

        object_name = self._sample_target_object_synonym(
            object_id=object_id, template_type=selected_type
        )
        instruction_options = deepcopy(self._instruction_options)
        if self._no_prefix_instruction_options:
            instruction_options.extend(self._no_prefix_instruction_options)

        verb = random.choice(instruction_options)
        template_values = {
            "verb": verb,
            "object": object_name,
            "color": attributes.color,
            "location": attributes.location,
        }
        instruction = selected_template.format(**template_values)

        # Allow a prefix if the selected verb is not part of the self._no_prefix_instruction_options
        if len(instruction_options) == len(self._instruction_options):
            allowed_prefix = True
        else:
            allowed_prefix = verb not in self._no_prefix_instruction_options
        if allowed_prefix and random.random() < 1 / 2:
            instruction = self._add_prefix(instruction, random.choice(self._prefix_options))
        return instruction.lower()

    def _add_prefix(self, instruction: str, prefix: str) -> str:
        return f"{prefix} {instruction}".lower()

    def _add_suffix(self, instruction: str, suffix: str) -> str:
        if instruction.endswith("."):
            instruction = instruction[:-1]
        return f"{instruction} {suffix}".lower()

    def _sample_target_object_synonym(self, object_id: str, template_type: str) -> str:
        object_name = get_object_readable_name_from_object_id(
            object_id=object_id,
            object_assets_to_names=self._assets_to_labels,
            special_name_cases=self._special_name_cases,
        )

        object_asset = get_object_asset_from_object_id(object_id, self._assets_to_labels)
        object_class = self._assets_to_labels[object_asset]

        # If it's not a `special case` object then the object class and the object readable name should be the same.
        # Therefore you can always sample a synonym.
        if object_name == object_class:
            object_name = random.choice(self.object_synonyms[object_asset])
        # If the template is not a verb_template we can use any synonym
        elif self._available_templates[template_type] != self._verb_templates:
            object_name = random.choice(self.object_synonyms[object_asset])
        return object_name


class GotoParaphraser(BaseParaphraser):
    """This is called in training only!"""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="goto")

        self._instruction_options = [
            "go to",
            "go back to",
            "go towards",
            "move to",
            "move closer to",
            "navigate to",
            "get closer to",
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

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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
            "fire",
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

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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
            "collect",
            "fetch",
            "get",
            "grab",
            "pick up",
            "take",
        ]

        self._available_templates = {
            "pickup": self._verb_templates,
            "pickup_color": self._verb_color_templates,
            "pickup_location": self._verb_location_templates,
        }

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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
            "leave the {pickable_object} in",
            "leave the {pickable_object} on",
            "place the {pickable_object} on",
            "place the {pickable_object} in",
            "put the {pickable_object} in",
            "put the {pickable_object} on",
            "put down the {pickable_object} on",
            "put the {pickable_object} down on",
            "insert the {pickable_object} in",
            "set the {pickable_object} at",
            "set the {pickable_object} on",
        ]

        self._available_templates = {
            "place": self._verb_templates,
            "place_color": self._verb_color_templates,
            "place_location": self._verb_location_templates,
        }

        self.requires_inventory = True

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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

        if inventory_object_id is None:
            raise AssertionError("PlaceParaphraser requires inventory.")
        pickable_object = random.choice(self.object_synonyms[inventory_object_id]).lower()
        instruction = instruction.format(pickable_object=pickable_object)
        return instruction


class BreakParaphraser(BaseParaphraser):
    """Paraphrase break instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="break")
        self._instruction_options = [
            "break",
            "break into pieces",
            "break to pieces",
            "crash",
            "crack",
            "shatter",
            "smash",
        ]
        augmented_prefix_options = [f"{opt} use the hammer to" for opt in self._prefix_options]
        self._prefix_options.extend(augmented_prefix_options)
        self._suffix_option = "with the hammer."

        self._available_templates = {
            "break": self._verb_templates,
            "break_color": self._verb_color_templates,
            "break_location": self._verb_location_templates,
        }
        self.requires_inventory = True

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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
        if proba < (1 / 3) and "hammer" not in instruction:
            instruction = self._add_suffix(instruction, self._suffix_option)
        return instruction


class CleanParaphraser(BaseParaphraser):
    """Paraphrase clean instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="clean")
        self._instruction_options = [
            "clean",
            "cleanse",
            "rinse",
            "soak",
            "sponge",
            "wash",
            "wipe",
        ]
        self._suffix_option = "in the sink."

        self._available_templates = {
            "clean": self._verb_templates,
        }
        self.requires_inventory = True

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
        """Get a clean instruction."""
        if inventory_object_id is None:
            raise AssertionError("CleanParaphraser requires inventory.")

        readable_name = get_object_readable_name_from_object_id(
            object_id=inventory_object_id,
            object_assets_to_names=self._assets_to_labels,
            special_name_cases=self._special_name_cases,
        )

        instruction = self._get_instruction(
            object_id=inventory_object_id,
            attributes=SimBotObjectAttributes(
                readable_name=readable_name,
            ),
            available_types=["clean"],
        )

        if random.random() < (1 / 2):
            instruction = self._add_suffix(instruction, self._suffix_option)
        return instruction


class PourParaphraser(BaseParaphraser):
    """Paraphrase pour instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="pour")
        self._instruction_options = [
            "pour {pourable_object} {preposition} the",
            "pour the {pourable_object} {preposition} the",
            "pour some {pourable_object} {preposition} the",
            "put {pourable_object} {preposition} the",
            "put the {pourable_object} {preposition} the",
            "put some {pourable_object} {preposition} the",
            "pour {pourable_object} from the {inventory_object} {preposition} the",
            "pour the {pourable_object} from the {inventory_object} {preposition} the",
            "pour some {pourable_object} from the {inventory_object} {preposition} the",
            "put {pourable_object} from the {inventory_object} {preposition} the",
            "put the {pourable_object} from the {inventory_object} {preposition} the",
            "put some {pourable_object} from the {inventory_object} {preposition} the",
        ]

        self._available_templates = {
            "pour": self._verb_templates,
            "pour_color": self._verb_color_templates,
            "pour_location": self._verb_location_templates,
        }
        self.requires_inventory = True
        self._pourable_inventory_mapping = {
            "Bowl_01": ["water", "milk", "cereal"],
            "Cereal_Box_01": ["cereal"],
            "CoffeeMug_Boss": ["water", "coffee"],
            "CoffeeMug_Yellow": ["water", "coffee"],
            "CoffeePot_01": ["water", "coffee"],
            "CoffeeBeans_01": ["coffee beans", "beans"],
            "MilkCarton_01": ["milk"],
        }
        self._prepositions = ["in", "into"]

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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
        if inventory_object_id is None:
            raise AssertionError("PourParaphraser requires inventory.")

        instruction_extra_slots = {
            "pourable_object": random.choice(
                self._pourable_inventory_mapping[inventory_object_id]
            ),
            "inventory_object": random.choice(self.object_synonyms[inventory_object_id]),
            "preposition": random.choice(self._prepositions),
        }
        instruction = instruction.format(**instruction_extra_slots)

        return instruction


class ScanParaphraser(BaseParaphraser):
    """Paraphrase scan instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="scan")
        self._instruction_options = ["scan", "examine", "survey", "study", "eye", "inspect"]

        self._available_templates = {
            "scan": self._verb_templates,
            "scan_color": self._verb_color_templates,
            "scan_location": self._verb_location_templates,
        }

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
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
        self._suffix_options = [
            "with water",
            "with water from the sink",
            "in the sink",
        ]

        self._available_templates = {
            "fill": self._verb_templates,
        }
        self.requires_inventory = True

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
        """Get a fill instruction."""
        if inventory_object_id is None:
            raise AssertionError("FillParaphraser requires inventory.")

        readable_name = get_object_readable_name_from_object_id(
            object_id=inventory_object_id,
            object_assets_to_names=self._assets_to_labels,
            special_name_cases=self._special_name_cases,
        )

        instruction = self._get_instruction(
            object_id=inventory_object_id,
            attributes=SimBotObjectAttributes(
                readable_name=readable_name,
            ),
            available_types=["fill"],
        )
        if random.random() < (1 / 2):
            instruction = self._add_suffix(instruction, random.choice(self._suffix_options))
        return instruction


class SearchParaphraser(BaseParaphraser):
    """Paraphrase search instructions."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        super().__init__(object_synonyms=object_synonyms, action_type="search")
        self._instruction_options = [
            "find",
            "locate",
            "search for",
            "look for",
            "seek",
            "trace",
            "investigate the room for",
            "explore the room for",
        ]
        self._no_prefix_instruction_options = [
            "where is",
            "do you see",
        ]

        self._available_templates = {
            "search": self._verb_templates,
            "search_color": self._verb_color_templates,
        }

    def __call__(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        inventory_object_id: Optional[str] = None,
    ) -> str:
        """Get a search instruction."""
        available_types = ["search"]
        object_color = attributes.color
        if object_color is not None:
            available_types.append("search_color")

        instruction = self._get_instruction(
            object_id=object_id, attributes=attributes, available_types=available_types
        )
        return instruction
