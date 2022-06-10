from typing import Any, Iterator

from overrides import overrides

from emma_datasets.datamodels import AnnotationType, DatasetName, TaskDescription
from emma_datasets.datamodels.datasets import AlfredMetadata
from emma_datasets.io import read_json
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class AlfredTaskDescriptionExtractor(AnnotationExtractor[TaskDescription]):
    """Split subgoal descriptions for ALFRED into multiple files."""

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.task_description

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.alfred

    def read(self, file_path: Any) -> list[dict[str, Any]]:
        """Read ALFRED metadata file."""
        return read_json(file_path)

    @overrides(check_signature=False)
    def convert(self, raw_feature: AlfredMetadata) -> list[TaskDescription]:
        """Convert raw feature to task description."""
        task_descriptions = []
        for ann in raw_feature.turk_annotations["anns"]:
            task_descriptions.append(TaskDescription(text=ann.task_desc))

        return task_descriptions

    def process_single_instance(self, raw_instance: dict[str, Any]) -> None:
        """Process raw instance and write to file."""
        structured_instance = AlfredMetadata.parse_obj(raw_instance)
        task_descriptions = self.convert(structured_instance)
        file_id = f"{structured_instance.task_id}"
        self._write(task_descriptions, file_id)

    def _read(self) -> Iterator[Any]:
        """Reads all the trajectory metadata from the train and valid_seen data paths.

        For ALFRED we have to override this to make sure that all the single trajectory files are
        correctly combined in a single list.
        """
        raw_data = (
            self.process_raw_file_return(self.read(file_path)) for file_path in self.file_paths
        )

        return self.postprocess_raw_data(raw_data)
