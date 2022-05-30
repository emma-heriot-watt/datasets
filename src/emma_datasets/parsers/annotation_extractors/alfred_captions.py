from typing import Any, Iterator

from overrides import overrides

from emma_datasets.datamodels import Caption, DatasetName
from emma_datasets.datamodels.datasets import AlfredMetadata
from emma_datasets.io import read_json
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class AlfredCaptionExtractor(AnnotationExtractor[Caption]):
    """Split subgoal descriptions for ALFRED into multiple files."""

    progress_bar_description = f"[b]Captions[/] from [u]{DatasetName.alfred.value}[/]"

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
                subgoal_captions.append(Caption(text=ann.high_descs[high_idx]))

            captions.append((high_idx, subgoal_captions))

        return captions

    def process_single_instance(self, raw_instance: dict[str, Any]) -> None:
        """Process raw instance and write to file."""
        structured_instance = AlfredMetadata.parse_obj(raw_instance)
        captions = self.convert(structured_instance)
        for high_idx, subgoal_captions in captions:
            caption_id = f"{structured_instance.task_id}_{high_idx}"
            self._write(subgoal_captions, caption_id)

    def _read(self) -> Iterator[Any]:
        """Reads all the trajectory metadata from the train and valid_seen data paths.

        For ALFRED we have to override this to make sure that all the single trajectory files are
        correctly combined in a single list.
        """
        raw_data = (
            self.process_raw_file_return(self.read(file_path)) for file_path in self.file_paths
        )

        return self.postprocess_raw_data(raw_data)
