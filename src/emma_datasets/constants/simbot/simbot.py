from functools import lru_cache
from typing import Any

from emma_datasets.common.settings import Settings
from emma_datasets.io import read_json


settings = Settings()

ARENA_JSON = settings.paths.constants.joinpath("simbot", "arena_definitions.json")
SYNTHETIC_JSON = settings.paths.constants.joinpath("simbot", "low_level_actions_templates.json")
OBJECT_SYNONYMS = settings.paths.constants.joinpath("simbot/object_id_synonyms.json")


@lru_cache(maxsize=1)
def get_arena_definitions() -> dict[str, Any]:
    """Load the arena definitions."""
    return read_json(ARENA_JSON)


@lru_cache(maxsize=1)
def get_low_level_action_templates() -> dict[str, Any]:
    """Load the low level action templates."""
    return read_json(SYNTHETIC_JSON)


@lru_cache(maxsize=1)
def get_objects_id_synonyms() -> dict[str, list[str]]:
    """Load the object synonyms."""
    return read_json(OBJECT_SYNONYMS)
