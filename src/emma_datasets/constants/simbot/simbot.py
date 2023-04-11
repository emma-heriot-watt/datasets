from collections import defaultdict
from functools import lru_cache
from typing import Any

from emma_datasets.common.settings import Settings
from emma_datasets.io import read_json


settings = Settings()

ARENA_JSON = settings.paths.constants.joinpath("simbot", "arena_definitions.json")
SYNTHETIC_JSON = settings.paths.constants.joinpath("simbot", "low_level_actions_templates.json")
OBJECT_ASSET_SYNONYMS_JSON = settings.paths.constants.joinpath("simbot/asset_synonyms.json")
CLASS_THRESHOLDS_JSON = settings.paths.constants.joinpath("simbot/class_thresholds2.json")
OBJECT_MANIFEST_JSON = settings.paths.constants.joinpath("simbot/ObjectManifest.json")


@lru_cache(maxsize=1)
def get_arena_definitions() -> dict[str, Any]:
    """Load the arena definitions."""
    return read_json(ARENA_JSON)


@lru_cache(maxsize=1)
def get_low_level_action_templates() -> dict[str, Any]:
    """Load the low level action templates."""
    return read_json(SYNTHETIC_JSON)


@lru_cache(maxsize=1)
def get_objects_asset_synonyms() -> dict[str, list[str]]:
    """Load the object synonyms."""
    return read_json(OBJECT_ASSET_SYNONYMS_JSON)


@lru_cache(maxsize=1)
def get_class_thresholds() -> dict[str, list[float]]:
    """Load the class thresholds."""
    return read_json(CLASS_THRESHOLDS_JSON)


@lru_cache(maxsize=1)
def get_object_manifest() -> dict[str, Any]:
    """Load the object manifest."""
    return read_json(OBJECT_MANIFEST_JSON)


@lru_cache(maxsize=1)
def get_pickable_objects_ids() -> list[str]:
    """Get all the pickable object ids from the object manifest."""
    object_manifest = get_object_manifest()

    pickable_object_ids = []
    for _, object_metadata in object_manifest.items():
        object_id = object_metadata["ObjectID"]
        if "PICKUPABLE" in object_metadata["ObjectProperties"]:
            pickable_object_ids.append(object_id)
    return pickable_object_ids


@lru_cache(maxsize=1)
def get_object_name_list_by_property(obj_property: str) -> list[str]:
    """Get a list of objects with a given property."""
    obj_manifest = get_object_manifest().values()
    obj_property = obj_property.upper()
    return [
        instance["ReadableName"].lower()
        for instance in obj_manifest
        if obj_property in instance["ObjectProperties"]
    ]


@lru_cache(maxsize=1)
def get_object_synonym(readable_name: str) -> list[str]:
    """Get the synonyms for an object."""
    labels_to_synonyms = defaultdict(list)
    arena_definitions = get_arena_definitions()
    assets_to_labels = arena_definitions["asset_to_label"]

    object_synonyms = get_objects_asset_synonyms()
    for object_asset, object_label in assets_to_labels.items():
        if object_asset == "Sticky Note":
            continue

        asset_synonyms = object_synonyms[object_asset]
        labels_to_synonyms[object_label.lower()].extend(asset_synonyms)
    return labels_to_synonyms[readable_name]
