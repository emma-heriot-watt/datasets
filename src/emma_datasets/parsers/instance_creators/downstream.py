from pathlib import Path
from typing import Any, TypeVar, Union

from rich.progress import Progress

from emma_datasets.datamodels import BaseInstance
from emma_datasets.db.storage import DataStorage, JsonStorage
from emma_datasets.parsers.instance_creators.generic import GenericInstanceCreator


InstanceModelType = TypeVar("InstanceModelType", bound=BaseInstance)


class DownstreamInstanceCreator(
    GenericInstanceCreator[Union[Path, str, dict[Any, Any]], InstanceModelType]
):
    """Create instances for downstream datasets.

    We assume that downstream datasets are either a Path to a file, or a string which can be parsed
    with orjson to a Pydantic model.
    """

    def __init__(
        self,
        instance_model_type: type[InstanceModelType],
        progress: Progress,
        task_description: str = "Creating instances",
        should_compress: bool = True,
        data_storage: DataStorage = JsonStorage(),  # noqa: WPS404
    ) -> None:
        super().__init__(
            progress=progress,
            task_description=task_description,
            should_compress=should_compress,
            data_storage=data_storage,
        )

        self.instance_model_type = instance_model_type

    def _create_instance(self, input_data: Union[Path, str, dict[Any, Any]]) -> InstanceModelType:
        """Parse the instance from the file and return it."""
        if isinstance(input_data, Path):
            return self.instance_model_type.parse_file(input_data)

        if isinstance(input_data, str):
            return self.instance_model_type.parse_raw(input_data)

        if isinstance(input_data, dict):
            return self.instance_model_type.parse_obj(input_data)

        raise NotImplementedError("Input data type is not supported.")
