import itertools
from typing import Any, Iterator

from pydantic import parse_obj_as

from emma_datasets.datamodels import Caption
from emma_datasets.datamodels.datasets import CocoCaption
from emma_datasets.parsers.instance_splitters.instance_splitter import InstanceSplitter


class CocoCaptionSplitter(InstanceSplitter[Caption]):
    """Split COCO captions into multiple files."""

    progress_bar_description = "Splitting captions for [u]COCO[/]"

    def process_raw_file_return(self, raw_data: Any) -> Any:
        """Only get the captions from the raw file."""
        return raw_data["annotations"]

    def postprocess_raw_data(self, raw_data: Any) -> Any:
        """Group the captions by image ID."""
        sorted_raw_data = sorted(raw_data, key=lambda k: k["image_id"])
        grouped_captions_generator = itertools.groupby(
            sorted_raw_data, key=lambda k: k["image_id"]
        )
        return (
            (image_id, list(grouped_captions))
            for image_id, grouped_captions in grouped_captions_generator
        )

    def convert(self, raw_feature: list[CocoCaption]) -> Iterator[Caption]:
        """Convert objects to the common Caption."""
        return (Caption(text=instance.caption) for instance in raw_feature)

    def process_single_instance(self, raw_instance: Any) -> None:
        """Process raw instance and write to file."""
        image_id, grouped_captions = raw_instance
        structured_raw = parse_obj_as(list[CocoCaption], grouped_captions)
        features = self.convert(structured_raw)
        self._write(features, image_id)
