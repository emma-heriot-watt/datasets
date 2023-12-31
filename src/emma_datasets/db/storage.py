import io
from abc import ABC, abstractmethod
from enum import Enum
from lzma import compress, decompress
from typing import Any

import orjson
from pydantic import BaseModel

from emma_datasets.common.logger import get_logger


logger = get_logger(__name__)

try:
    import torch  # noqa: WPS433
except ImportError:
    logger.warning(
        "Unable to import `torch`. You will NOT be able to use the `TorchDataStorage` class. "
        + "Consider installing it if you want to use it!"
    )


class StorageType(Enum):
    """Different serialisation formats for objects in SQLite database."""

    torch = "torch"
    json = "json"


class DataStorage(ABC):
    """Abstract class for converting data object to bytes for the SQLite database.

    Data are by default stored as BLOB type in the database.
    """

    storage_type: StorageType

    @abstractmethod
    def decompress(self, data_buf: bytes) -> Any:
        """Given a byte representation of an object, returns the original object representation."""
        raise NotImplementedError

    @abstractmethod
    def compress(self, data: Any) -> bytes:
        """Given an object representation, returns a compressed byte representation."""
        raise NotImplementedError


class JsonStorage(DataStorage):
    """Uses orjson serialisation to convert Python object to bytes."""

    storage_type: StorageType = StorageType.json

    def decompress(self, data_buf: bytes) -> Any:
        """Decompress using LZMA and then loads the underlying bytes using orjson."""
        return orjson.loads(decompress(data_buf))

    def compress(self, data: Any) -> bytes:
        """Uses orjson + LZMA compression to generate a byte representation of the object.

        If the JSON data has already been compressed into bytes, then it will bypass the
        compression and return it quick.
        """
        if isinstance(data, bytes):
            return data

        return compress(
            orjson.dumps(
                data,
                default=self._serialize_custom_types,
                option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
            )
        )

    def _serialize_custom_types(self, custom_obj: Any) -> Any:
        if isinstance(custom_obj, BaseModel):
            return custom_obj.json(by_alias=True)

        raise TypeError


class TorchStorage(DataStorage):
    """Data storage that uses the PyTorch Pickle format for serialising Python objects."""

    storage_type: StorageType = StorageType.torch

    def decompress(self, data_buf: bytes) -> Any:
        """Loads an object from a pytorch-pickle representation."""
        buffer = io.BytesIO(data_buf)

        return torch.load(buffer)

    def compress(self, data: Any) -> bytes:
        """Given an object, returns its byte representation using pytorch-pickle."""
        buffer = io.BytesIO()

        torch.save(data, buffer)
        buffer.seek(0)

        return buffer.read()
