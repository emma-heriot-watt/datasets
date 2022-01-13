import orjson
from pydantic import BaseModel as PydanticBaseModel

from src.io.json import orjson_dumps


class BaseModel(PydanticBaseModel):
    """Base model class, inherited from Pydantic."""

    class Config:  # noqa: WPS431
        """Updated config."""

        json_loads = orjson.loads
        json_dumps = orjson_dumps
        arbitrary_types_allowed = True
