import orjson
from pydantic import BaseModel as PydanticBaseModel

from emma_datasets.io.json import orjson_dumps


class BaseModel(PydanticBaseModel):
    """Base model class, inherited from Pydantic."""

    class Config:
        """Updated config."""

        json_loads = orjson.loads
        json_dumps = orjson_dumps
        arbitrary_types_allowed = True
