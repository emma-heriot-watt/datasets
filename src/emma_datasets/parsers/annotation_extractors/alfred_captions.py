from collections.abc import Iterator
from typing import Any

from overrides import overrides

from emma_datasets.datamodels import AnnotationType, Caption, DatasetName
from emma_datasets.datamodels.datasets import AlfredMetadata
from emma_datasets.io import read_json
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class AlfredCaptionExtractor(AnnotationExtractor[Caption]):
    """Split subgoal descriptions for ALFRED into multiple files."""

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.caption

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.alfred

    def read(self, file_path: Any) -> list[dict[str, Any]]:
        """Read ALFRED metadata file."""
        return read_json(file_path)

    @overrides(check_signature=False)
    def convert(self, raw_feature: AlfredMetadata) -> list[tuple[int, list[Caption]]]:
        """Convert raw feature to caption."""
        num_subgoals = min(len(ann.high_descs) for ann in raw_feature.turk_annotations["anns"])

        captions = []
        for high_idx in range(num_subgoals):
            subgoal_captions = []
            for ann in raw_feature.turk_annotations["anns"]:
                subgoal_captions.append(Caption(text=self._prep_caption(ann.high_descs[high_idx])))
            captions.append((high_idx, subgoal_captions))

        return captions

    def process_single_instance(self, raw_instance: dict[str, Any]) -> None:
        """Process raw instance and write to file."""
        structured_instance = AlfredMetadata.parse_obj(raw_instance)
        self._process_subgoal_instances(structured_instance)
        self._process_trajectory_instance(structured_instance)

    def _process_subgoal_instances(self, structured_instance: AlfredMetadata) -> None:
        """Process descriptions for each subgoal and write to file."""
        captions = self.convert(structured_instance)
        for high_idx, subgoal_captions in captions:
            caption_id = f"{structured_instance.task_id}_{high_idx}"
            self._write(subgoal_captions, caption_id)

    def _process_trajectory_instance(self, structured_instance: AlfredMetadata) -> None:
        """Merge descrptions for all subgoals and write to file."""
        captions = self._merge_high_descs(structured_instance)
        self._write(captions, f"{structured_instance.task_id}")

    def _prep_caption(self, caption: str) -> str:
        """Make sure captions end with full stop.

        This is so that we can mask full sentences for pretraining.
        """
        caption = caption.rstrip()
        if not caption.endswith("."):
            caption = f"{caption}."
        return caption

    def _merge_high_descs(self, raw_feature: AlfredMetadata) -> list[Caption]:
        """Merge high level descriptions in a single caption."""
        captions = []

        for ann in raw_feature.turk_annotations["anns"]:
            ann_captions = [self._prep_caption(caption) for caption in ann.high_descs]
            captions.append(Caption(text=" ".join(ann_captions)))

        return captions

    def _read(self) -> Iterator[Any]:
        """Reads all the trajectory metadata from the train and valid_seen data paths.

        For ALFRED we have to override this to make sure that all the single trajectory files are
        correctly combined in a single list.
        """
        raw_data = (
            self.process_raw_file_return(self.read(file_path)) for file_path in self.file_paths
        )

        return self.postprocess_raw_data(raw_data)
