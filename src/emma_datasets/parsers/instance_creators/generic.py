import itertools
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
        self._storage = data_storage

    def __call__(
        self,
        input_data: Iterable[InputType],
        progress: Progress,
        pool: Optional[Pool] = None,
    ) -> Union[Iterator[OutputType], Iterator[bytes]]:
        """Create instances from a list of input data."""
        progress.reset(self.task_id, start=True, visible=True)

        if pool is not None:
            iterator = pool.imap_unordered(self.create_instances, input_data)
            for instances in iterator:
                progress.advance(self.task_id, advance=len(instances))
                yield from itertools.chain(instances)

        else:
            for scene in input_data:
                scene_instances = self.create_instances(scene)
                progress.advance(self.task_id, advance=len(scene_instances))
                yield from itertools.chain(scene_instances)

    def create_instances(self, input_data: InputType) -> Union[list[OutputType], list[bytes]]:
        """Create all the possible instances from a single piece of input data.

        If desired, also compress the instance into the bytes representation to faciliate drastic
        speed increases in writing to the DB.
        """
        instances = self._create_instances(input_data)

        if self._should_compress:
            return [self._storage.compress(instance) for instance in instances]

        return instances

    @abstractmethod
    def _create_instances(self, input_data: InputType) -> list[OutputType]:
        """The main logic for creating the instance goes here."""
        raise NotImplementedError
