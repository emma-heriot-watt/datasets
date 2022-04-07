from pathlib import Path

from emma_datasets.datamodels import TeachEdhInstance
from emma_datasets.parsers.instance_creators.generic import GenericInstanceCreator


class TeachEdhInstanceCreator(GenericInstanceCreator[Path, TeachEdhInstance]):
    """Create TEACh EDH Instances from the paths."""

    def _create_instance(self, input_data: Path) -> TeachEdhInstance:
        """Parse the instance from the file and return it."""
        return TeachEdhInstance.parse_file(input_data)
