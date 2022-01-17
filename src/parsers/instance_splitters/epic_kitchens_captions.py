from typing import Any

from src.datamodels import Caption, DatasetName
from src.datamodels.datasets import EpicKitchensNarrationMetadata
from src.io import read_csv
from src.parsers.instance_splitters.instance_splitter import InstanceSplitter


class EpicKitchensCaptionSplitter(InstanceSplitter[Caption]):
    """Split captions for EpicKitchens into multiple files."""

    progress_bar_description = f"[b]Captions[/] from [u]{DatasetName.epic_kitchens.value}[/]"

    def read(self, file_path: Any) -> list[dict[str, Any]]:
        """Read Epic Kitchen CSV file."""
        return read_csv(file_path)

    def convert(self, raw_feature: EpicKitchensNarrationMetadata) -> list[Caption]:
        """Convert raw feature to caption."""
        return [Caption(text=raw_feature.narration)]

    def process_single_instance(self, raw_instance: dict[str, Any]) -> None:
        """Process raw instance and write to file."""
        structured_instance = EpicKitchensNarrationMetadata.parse_obj(raw_instance)
        caption = self.convert(structured_instance)
        self._write(caption, structured_instance.narration_id)
