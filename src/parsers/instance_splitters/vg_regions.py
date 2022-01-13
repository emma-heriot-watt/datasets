from typing import Any, Iterator

import numpy

from src.datamodels import Region
from src.datamodels.datasets import VgImageRegions
from src.parsers.instance_splitters.instance_splitter import InstanceSplitter


class VgRegionsSplitter(InstanceSplitter[VgImageRegions, Region]):
    """Split Regions per VG instance into multiple files."""

    progress_bar_description = "Splitting regions for [u]Visual Genome[/]"

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
