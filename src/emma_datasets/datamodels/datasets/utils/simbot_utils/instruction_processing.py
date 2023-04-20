import re
from typing import Any, Optional

import spacy

from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotClarificationTypes,
)


def get_object_asset_from_object_id(object_id: str, object_assets_to_names: dict[str, str]) -> str:
    """Map the object id to its object asset.

    Example:
        (object_asset, object_name) = (V_Monitor_Laser_1000, V_Monitor_Laser)
    """
    object_assets = object_assets_to_names.keys()
    # Case1: Object id in action matches exactly with object assets
    if object_id in object_assets:
        return object_id

    # Case2: The object id contains a substring that matches with the object assests
    # Example: Desk_01_1000
    # Because the ids can have additional tags we need to remove these tags
    # and check if they asset after removing the tags match an object asset
    object_id_components = object_id.split("_")

    for idx in range(len(object_id_components), 0, -1):
        # tries to match the longest sub-string first
        object_name_candidate = "_".join(object_id_components[:idx])
        if object_name_candidate in object_assets:
            return object_name_candidate
    return object_id


def get_object_label_from_object_id(object_id: str, object_assets_to_names: dict[str, str]) -> str:
    """Map the object id for a given action to its name.

    The name corresponds to the object detection label. Example:
        (object_id, object_name) = (V_Monitor_Laser_1000, Computer)
    """
    # Case1: Object asset in action matches exactly with object assets
    object_name_candidate = object_assets_to_names.get(object_id, None)
    if object_name_candidate is not None:
        return object_name_candidate

    # Case2: The object asset in action contains a substring that matches with the object assests
    # Example: Desk_01_1000
    # Because the assets can have additional tags we need to remove these tags
    # and check if they asset after removing the tags match an object label
    object_asset_components = object_id.split("_")

    for idx in range(len(object_asset_components), 0, -1):
        # tries to match the longest sub-string first
        object_name_candidate = "_".join(object_asset_components[:idx])
        object_name_candidate = object_assets_to_names.get(object_name_candidate, None)
        if object_name_candidate is not None:
            return object_name_candidate

    return object_id


def get_object_readable_name_from_object_id(
    object_id: str, object_assets_to_names: dict[str, str], special_name_cases: dict[str, str]
) -> str:
    """Map the object asset for a given action to its readable name.

    Example:
        (object_asset, object_name) = (V_Monitor_Laser_1000, Laser Monitor)
    """
    object_asset = get_object_asset_from_object_id(object_id, object_assets_to_names)
    readable_name = special_name_cases.get(object_asset, None)
    if readable_name is None:
        return object_assets_to_names.get(object_asset, object_asset)
    return readable_name


class ClarificationTargetExtractor:
    """Extract the target noun phrase for the clarfication question.

    Spelling correction did not work for some cases, for which we fix it manually.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:

        self.nlp = spacy.load(spacy_model)

        self.nlp.add_pipe("merge_noun_chunks")
        self._prefer_naive = {"look"}  # The verb 'look' is sometimes confused as a noun
        self._skipped_nouns = {"can", "sink", "floppy"}  # Cannot identify certain words as nouns
        # Rule-based approach to get the target word from its position.
        # This fails for compound words, for which we use spacy noun chunks.
        self.target_index = {
            SimBotClarificationTypes.description: 3,
            SimBotClarificationTypes.disambiguation: 1,
            SimBotClarificationTypes.location: 3,
        }
        # This is a manual fix for mapping common target names to object detector labels
        self._normalized_name_map = {
            "changer": "color changer",
            "swapper": "color changer",
            "swapper machine": "color changer",
            "panel": "control panel",
            "mschine": "machine",
            "refrigeratorand": "fridge",
            "control": "control panel",
            "ray": "freeze ray",
            "maker": "coffee maker",
            "refrigerator": "fridge",
            "tip": "laser tip",
            "pot": "coffee pot",
            "floppy": "floppy disk",
            "cartridge": "printer cartridge",
            "catridge": "printer cartridge",
            "cartrige": "printer cartridge",
            "desk": "table",
            "jelly": "jar",
            "monitor": "computer",
            "extinguisher": "fire extinguisher",
            "figure": "action figure",
            "coffeemaker": "coffee maker",
            "unmaker": "coffee unmaker",
            "toast": "bread",
            "loaf": "bread",
            "pc": "computer",
            "terminal": "computer",
            "freeze ray controller": "computer",
            "bean": "coffee beans",
            "cereal": "cereal box",
            "driver": "screwdriver",
            "disk": "floppy disk",
            "disc": "floppy disk",
            "faucet": "sink",
            "tap": "sink",
            "platform": "wall shelf",
            "cupboard": "drawer",
            "jug": "coffee pot",
            "soda": "can",
            "pipe": "sink",
            "sign": "warning sign",
            "countertop": "counter top",
            "oven": "microwave",
            "saw": "handsaw",
            "hammmer": "hammer",
            "candy": "candy bar",
        }

        self._nomalize_types = {
            "machine": {
                "time": "time machine",
                "coffee": "coffee maker",
                "laser": "laser",
                "freeze ray": "freeze ray",
                "color": "color changer",
                "print": "printer",
            },
            "slice": {
                "apple": "apple",
                "cake": "cake",
                "pie": "pie",
                "bread": "bread",
                "toast": "bread",
            },
            "button": {
                "red": "button",
                "blue": "button",
                "green": "button",
            },
            "target": {
                "freeze ray": "wall shelf",
                "laser": "wall shelf",
            },
            "container": {"milk": "milk"},
        }
        self._normalize_synonyms = {
            "machine": {"machine", "station"},
            "target": {"target", "shelf"},
            "slice": {"slice"},
            "button": {"button"},
            "container": {"container"},
        }
        self._object_classes = get_arena_definitions()["asset_to_label"].values()

    def __call__(
        self,
        question: str,
        question_type: SimBotClarificationTypes,
    ) -> Optional[str]:
        """Preprocess the clarification target."""
        tokens = question.split()
        target_index = min(self.target_index[question_type], len(tokens) - 1)
        naive_target = self.get_naive_target(tokens, target_index=target_index)
        target = self.get_target(question, target_index=target_index)
        if target is None or naive_target in self._prefer_naive:
            target = naive_target

        return target

    def normalize_target(self, target: Optional[str], instruction: str) -> Optional[str]:
        """Convert the target to an object detection label."""
        if target is None:
            return target
        # First, use a list of manual common mappings
        normalized_target = self._normalized_name_map.get(target, target)
        # Second, for a number of categories
        for object_type in self._nomalize_types:
            normalized_target = self._normalized_names(normalized_target, instruction, object_type)

        normalized_target = normalized_target.title()
        return normalized_target

    def get_naive_target(self, question_tokens: list[str], target_index: int) -> str:
        """Get the target based on the word position."""
        naive_target = question_tokens[target_index]
        return re.sub(r"[^\w\s]", "", naive_target)

    def get_target(self, question: str, target_index: int) -> Optional[str]:  # noqa: WPS231
        """Apply spell correction and find a noun phrase."""
        doc = self.nlp(question.lower())
        target = None
        for index, token in enumerate(doc):
            if index > target_index and token.is_stop:
                continue
            if token.tag_ in {"NNP", "NN"}:
                target = token.text.replace("which ", "")
                target = target.replace("the ", "")
            elif index == target_index and token.text in self._skipped_nouns:
                target = token.text
            if target is not None:
                break

        return target

    def _normalized_names(self, normalized_target: str, instruction: str, object_type: str) -> str:
        if normalized_target in self._normalize_synonyms[object_type]:
            normalized_types = self._nomalize_types[object_type]
            for keyword, normalized_name in normalized_types.items():
                if keyword in instruction:
                    return normalized_name
        return normalized_target


class InventoryObjectfromTrajectory:
    """Add the inventory object to the actions."""

    def __init__(self) -> None:
        self._object_assets_to_names = get_arena_definitions()["asset_to_label"]

    def __call__(
        self, actions: list[dict[str, Any]], initial_inventory: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Add the inventory object to actions."""
        inventory_object = initial_inventory
        for action in actions:
            action["inventory_object_id"] = inventory_object
            # Update the object that will be held after the current action
            if action["type"] == "Pickup":
                inventory_object = action["pickup"]["object"]["id"]

            elif action["type"] == "Place":
                inventory_object = None
        return actions
