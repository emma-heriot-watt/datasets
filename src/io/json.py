from pathlib import Path
from typing import Any, Optional, Union

import orjson


DEFAULT_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE


def read_json(path: Union[str, Path]) -> Any:
    """Read JSON file and return."""
    with open(path) as json_file:
        data = orjson.loads(json_file.read())
    return data


def write_json(path: Union[str, Path], data: Any) -> None:
    """Write any data to a JSON file."""
    with open(path, "wb") as save_file:
        save_file.write(orjson.dumps(data, option=DEFAULT_OPTIONS))


def orjson_dumps(v: Any, *, default: Optional[Any]) -> str:
    """Convert Model to JSON string.

    orjson.dumps returns bytes, to match standard json.dumps we need to decode.
    """
    return orjson.dumps(v, default=default, option=DEFAULT_OPTIONS).decode()
