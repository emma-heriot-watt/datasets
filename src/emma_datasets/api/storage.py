import io
import os
import sqlite3
from abc import ABC, abstractmethod
from enum import Enum
from lzma import compress, decompress
from types import TracebackType
from typing import Any, Iterable, Optional, Union

import orjson

from emma_datasets.common.logger import get_logger


logger = get_logger(__name__)

try:
    import torch  # noqa: WPS433
except ImportError:
    logger.warning(
        "Unable to import `torch`. You will NOT be able to use the `TorchDataStorage` class. "
        + "Consider installing it if you want to use it!"
    )

CREATE_DATA_TABLE = """
    CREATE TABLE dataset (data_id INTEGER PRIMARY KEY, example_id TEXT, data BLOB);
"""

INSERT_DATA_TABLE_FORMAT = """
    INSERT INTO dataset VALUES (?, ?, ?);
"""

SELECT_INDEX_FORMAT = """
    SELECT data FROM dataset WHERE data_id = ?;
"""

SELECT_EXID_FORMAT = """
    SELECT data FROM dataset WHERE example_id = ?;
"""

COUNT_INSTANCES = """
    SELECT COUNT(data_id) from dataset;
"""

DROP_TABLE = """
    DROP TABLE IF EXISTS dataset;
"""

CREATE_ID_INDEX = """
    CREATE INDEX id_index ON dataset (data_id);
"""

CREATE_EXID_INDEX = """
    CREATE INDEX example_id_index ON dataset (example_id);
"""

DELETE_EXAMPLES = """
    DELETE FROM dataset WHERE example_id = ?;
"""


class StorageType(Enum):
    """Different serialisation formats for objects in SQLite database."""

    torch = "torch"
    json = "json"


class DataStorage(ABC):
    """Abstract class for converting data object to bytes for the SQLite database.

    Data are by default stored as BLOB type in the database.
    """

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

    def decompress(self, data_buf: bytes) -> Any:
        """Decompress using LZMA and then loads the underlying bytes using orjson."""
        return orjson.loads(decompress(data_buf))

    def compress(self, data: Any) -> bytes:
        """Uses orjson + LZMA compression to generate a byte representation of the object."""
        return compress(
            orjson.dumps(data, option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY)
        )


class TorchStorage(DataStorage):
    """Data storage that uses the PyTorch Pickle format for serialising Python objects."""

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


DataStorage.register(TorchStorage)
DataStorage.register(JsonStorage)


class DatasetDB:
    """A class that mimics the dict interface to access to an SQLite database storing a dataset."""

    def __init__(
        self,
        db_dir: str,
        readonly: bool = True,
        batch_size: int = 512,
        storage_type: StorageType = StorageType.json,
    ):
        """Instantiates an object that can be used to manipulate an SQLite database."""
        self.readonly = readonly
        self.db_dir = db_dir
        if self.readonly and not os.path.exists(self.db_dir):
            raise ValueError(
                f"You specified a <read-only> option but the path to the DB doesn't exist!\nDatabase path: {self.db_dir}"
            )
        self._batch_size = batch_size
        self._storage_type = self._get_storage_type(storage_type)

        self._env: sqlite3.Connection
        self._write_count: int = 0
        self._cache: list[Any] = []

    def iterkeys(self) -> Iterable[tuple[int, str]]:
        """Returns an iterator over the keys of the dataset."""
        self.open()

        yield from (
            (data_id, example_id)
            for data_id, example_id in self._env.execute("SELECT data_id, example_id FROM dataset")
        )

    def open(self) -> None:
        """Opens the connection to the database, if it's not open already."""
        if not self._is_open:
            self._open()

    def close(self) -> None:
        """Closes the underlying connection with the SQLite database."""
        if self._is_open:
            if self._write_count > 0:
                with self._env:
                    self._env.executemany(INSERT_DATA_TABLE_FORMAT, self._cache)
                self._cache.clear()
                self._write_count = 0

            self._env.close()

    def flush(self) -> None:
        """Finalises the serialisation of the cached instances to the database."""
        if self._write_count > 0:
            # there are some pending transactions that have to be committed
            with self._env:
                self._env.executemany(INSERT_DATA_TABLE_FORMAT, self._cache)

            self._cache.clear()
            self._write_count = 0

    def update(self, data_id: int, example_id: str) -> None:
        """Updates the data_id column with a new one for the example."""
        with self._env:
            self._env.execute(
                "UPDATE dataset SET data_id = ? WHERE example_id = ?", (data_id, example_id)
            )

    def __enter__(self) -> "DatasetDB":
        """Returns the current database when context manager is initialised."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Closes the connection to the database when exiting the context manager."""
        self.close()

    def __del__(self) -> None:  # noqa: WPS603
        """Closes the connection to the database when object goes out of scope."""
        self.close()

    def __iter__(self) -> Iterable[tuple[int, str, Any]]:
        """Iterator over the instances of the database."""
        self.open()

        yield from (
            (data_id, example_id, self._storage_type.decompress(data_buf))
            for data_id, example_id, data_buf in self._env.execute("SELECT * FROM dataset")
        )

    def __contains__(self, key: Union[int, tuple[int, str]]) -> bool:
        """Verifies whether a given key is contained in the dataset."""
        self.open()

        if isinstance(key, int):
            # in this case we assume we're using directly an index
            query_format = SELECT_INDEX_FORMAT
        else:
            query_format = SELECT_EXID_FORMAT

        db_result = self._env.execute(query_format, (str(key),))

        return db_result.fetchone() is not None

    def __getitem__(self, key: Union[int, tuple[int, str]]) -> Any:
        """Returns the object associated with a given key."""
        self.open()

        if isinstance(key, int):
            # in this case we assume we're using directly an index
            query_format = SELECT_INDEX_FORMAT
        else:
            query_format = SELECT_EXID_FORMAT

        db_result = self._env.execute(query_format, (str(key),))

        db_item = None
        for res in db_result:
            db_item = res[0]

        if db_item is None:
            raise KeyError(f"No record for key: '{key}'")

        return self._storage_type.decompress(db_item)

    def __len__(self) -> int:
        """Returns the number of instances in the database."""
        self.open()

        self.flush()
        res_it = self._env.execute(COUNT_INSTANCES)

        res = next(res_it)

        return res[0] if res is not None else 0

    def __delitem__(self, key: tuple[int, str]) -> None:  # noqa: WPS603
        """Removes an instance from the database having a specific key."""
        self.open()

        # there are some pending transactions that have to be committed
        with self._env:
            self._env.execute(DELETE_EXAMPLES, (key,))

    def __setitem__(self, key: tuple[int, str], db_value: Any) -> None:
        """Inserts a new instance in the database using the specified (key, value)."""
        self.open()

        if self.readonly:
            raise ValueError("readonly text DB")

        data_id, example_id = key

        data = self._storage_type.compress(db_value)

        self._write_count += 1
        self._cache.append((data_id, example_id, data))

        if self._write_count > 1 and self._write_count % self._batch_size == 0:
            self.flush()

    def _get_storage_type(self, storage_type: StorageType) -> DataStorage:
        """Returns the data storage used to serialise the instances of this dataset."""
        if storage_type == StorageType.json:
            return JsonStorage()

        if storage_type == StorageType.torch:
            return TorchStorage()

        raise NotImplementedError(f"Invalid data storage type: {storage_type}")

    def _create_tables(self) -> None:
        """Generates the underlying database tables, if they don't exist."""
        with self._env:
            self._env.execute(DROP_TABLE)

        with self._env:
            self._env.execute(CREATE_DATA_TABLE)

    def _create_indexes(self) -> None:
        """Creates database indexes on the data_id and example_id columns."""
        with self._env:
            self._env.execute(CREATE_ID_INDEX)

        with self._env:
            self._env.execute(CREATE_EXID_INDEX)

    @property
    def _is_open(self) -> bool:
        """Checks whether the connection with the database is open."""
        try:
            self._env.cursor()
            return True
        except Exception:
            return False

    def _open(self) -> None:
        """Opens the connection to the underlying SQLite database."""
        self._env = sqlite3.connect(self.db_dir)

        if self.readonly:
            # training
            self._write_count = 0
            self._cache = []
        else:
            # prepro
            self._write_count = 0
            self._cache = []
            self._create_tables()
            self._create_indexes()
