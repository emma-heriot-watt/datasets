from abc import ABC, abstractmethod
from multiprocessing.pool import Pool
from typing import Generic, Iterable, Iterator, Optional, TypeVar, Union

from pydantic import BaseModel
from rich.progress import Progress

from emma_datasets.db import DataStorage, JsonStorage


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType", bound=BaseModel)


class GenericInstanceCreator(ABC, Generic[InputType, OutputType]):
    """Create instances from groups of metadata from all the datasets."""

    def __init__(
        self,
        progress: Progress,
        task_description: str = "Creating instances",
        data_storage: DataStorage = JsonStorage(),  # noqa: WPS404
        should_compress: bool = False,
    ) -> None:
        self.task_id = progress.add_task(
            task_description,
            visible=False,
            start=False,
            total=float("inf"),
            comment="",
        )

        self._should_compress = should_compress
        self.storage = data_storage

    def __call__(
        self,
        input_data: Iterable[InputType],
        progress: Progress,
        pool: Optional[Pool] = None,
    ) -> Union[Iterator[OutputType], Iterator[bytes]]:
        """Create instances from a list of input data."""
        progress.start_task(self.task_id)
        progress.update(self.task_id, visible=True)

        iterator: Iterator[Union[OutputType, bytes]]

        if pool is not None:
            iterator = pool.imap_unordered(self.create_instance, input_data)
        else:
            iterator = (self.create_instance(instance) for instance in input_data)

        for instance in iterator:
            progress.advance(self.task_id)
            yield instance

    def create_instance(self, input_data: InputType) -> Union[OutputType, bytes]:
        """Create the instance from a single piece of input data.

        If desired, also compress the instance into the bytes representation to faciliate drastic
        speed increases in writing to the DB.
        """
        instance = self._create_instance(input_data)

        if self._should_compress:
            return self.storage.compress(instance)

        return instance

    @abstractmethod
    def _create_instance(self, input_data: InputType) -> OutputType:
        """The main logic for creating the instance goes here."""
        raise NotImplementedError
