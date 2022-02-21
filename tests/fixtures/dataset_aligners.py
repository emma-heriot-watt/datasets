from pytest_cases import fixture, parametrize
from rich.progress import Progress

from emma_datasets.datamodels.datasets import CocoImageMetadata, GqaImageMetadata, VgImageMetadata
from emma_datasets.parsers.dataset_aligner import DatasetAligner
from emma_datasets.parsers.dataset_metadata import (
    CocoMetadataParser,
    GqaMetadataParser,
    VgMetadataParser,
)


@fixture
def vg_coco_aligner(
    vg_metadata_parser: VgMetadataParser,
    coco_metadata_parser: CocoMetadataParser,
    progress: Progress,
) -> DatasetAligner[VgImageMetadata, CocoImageMetadata]:
    return DatasetAligner[VgImageMetadata, CocoImageMetadata](
        vg_metadata_parser,
        coco_metadata_parser,
        source_mapping_attr_for_target="coco_id",
        target_mapping_attr_for_source="id",
        progress=progress,
    )


@fixture
def gqa_vg_aligner(
    vg_metadata_parser: VgMetadataParser,
    gqa_metadata_parser: GqaMetadataParser,
    progress: Progress,
) -> DatasetAligner[GqaImageMetadata, VgImageMetadata]:
    return DatasetAligner[GqaImageMetadata, VgImageMetadata](
        gqa_metadata_parser,
        vg_metadata_parser,
        source_mapping_attr_for_target="id",
        target_mapping_attr_for_source="image_id",
        progress=progress,
    )


@fixture
@parametrize("dataset_aligner", [vg_coco_aligner, gqa_vg_aligner])
def dataset_aligner(dataset_aligner):
    return dataset_aligner
