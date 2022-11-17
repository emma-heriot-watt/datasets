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
