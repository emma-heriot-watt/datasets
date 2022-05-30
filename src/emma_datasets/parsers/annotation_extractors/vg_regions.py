from typing import Any, Iterator

import numpy

from emma_datasets.datamodels import Region
from emma_datasets.datamodels.datasets import VgImageRegions
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class VgRegionsExtractor(AnnotationExtractor[Region]):
    """Split Regions per VG instance into multiple files."""

    progress_bar_description = "[b]Regions[/] from [u]Visual Genome[/]"

    def convert(self, raw_feature: VgImageRegions) -> Iterator[Region]:
        """Convert raw region description to a Region instance."""
        yield from (
            Region(
                caption=raw_region.phrase,
                bbox=numpy.array(
                    [
                        raw_region.x,
                        raw_region.y,
                        raw_region.width,
                        raw_region.height,
                    ],
                    dtype=numpy.float32,
                ),
            )
            for raw_region in raw_feature.regions
        )

    def process_single_instance(self, raw_feature: Any) -> None:
        """Process raw instance and write Region to file."""
        structured_raw = VgImageRegions.parse_obj(raw_feature)
        features = self.convert(structured_raw)
        self._write(features, structured_raw.id)
