from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import orjson
from pydantic import BaseModel as PydanticBaseModel

from emma_datasets.datamodels.constants import MediaType
from emma_datasets.io.json import orjson_dumps


class BaseModel(PydanticBaseModel):
    """Base model class, inherited from Pydantic."""

    class Config:
        """Updated config."""

        json_loads = orjson.loads
        json_dumps = orjson_dumps
        arbitrary_types_allowed = True


class BaseInstance(BaseModel, ABC):
    """Base instance class with common attributes and method used by all instances."""

    @property
    @abstractmethod
    def modality(self) -> MediaType:
        """Returns the modality of the instance."""
        raise NotImplementedError

    @property
    @abstractmethod
    def features_path(self) -> Union[Path, list[Path]]:
        """Get the path to the features for this instance."""
        raise NotImplementedError
