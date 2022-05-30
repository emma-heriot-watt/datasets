import itertools
from typing import Iterator

from pytest_cases import fixture
from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName
from emma_datasets.datamodels.datasets import CocoImageMetadata, GqaImageMetadata, VgImageMetadata
from emma_datasets.parsers.align_multiple_datasets import AlignMultipleDatasets
from emma_datasets.parsers.dataset_aligner import DatasetAligner
from emma_datasets.parsers.dataset_metadata import AlfredMetadataParser, EpicKitchensMetadataParser


@fixture
def vg_coco_gqa_grouped_metadata(
    vg_coco_aligner: DatasetAligner[VgImageMetadata, CocoImageMetadata],
    gqa_vg_aligner: DatasetAligner[GqaImageMetadata, VgImageMetadata],
    progress: Progress,
) -> Iterator[list[DatasetMetadata]]:
    align_multiple_datasets = AlignMultipleDatasets(DatasetName.visual_genome, progress)

    aligned_metadata_iterable = [
        vg_coco_aligner.get_aligned_metadata(),
        gqa_vg_aligner.get_aligned_metadata(),
    ]

    all_metadata_groups = align_multiple_datasets(*aligned_metadata_iterable)

    return all_metadata_groups


@fixture
def epic_kitchens_grouped_metadata(
    epic_kitchens_metadata_parser: EpicKitchensMetadataParser, progress: Progress
) -> Iterator[list[DatasetMetadata]]:
    return (
        [epic_kitchens_metadata_parser.convert_to_dataset_metadata(metadata)]
        for metadata in epic_kitchens_metadata_parser.get_metadata(progress)
    )


@fixture
def alfred_grouped_metadata(
    alfred_metadata_parser: AlfredMetadataParser, progress: Progress
) -> Iterator[list[DatasetMetadata]]:
    alfred_metadata = alfred_metadata_parser.get_metadata(progress)
    dataset_metadata_iterator = itertools.chain.from_iterable(
        alfred_metadata_parser.convert_to_dataset_metadata(metadata)
        for metadata in alfred_metadata
    )
    dataset_metadata = ([metadata] for metadata in dataset_metadata_iterator)

    return dataset_metadata


@fixture
def all_grouped_metadata(
    vg_coco_gqa_grouped_metadata: Iterator[list[DatasetMetadata]],
    epic_kitchens_grouped_metadata: Iterator[list[DatasetMetadata]],
    alfred_grouped_metadata: Iterator[list[DatasetMetadata]],
    all_extracted_annotations: bool,
) -> Iterator[list[DatasetMetadata]]:
    assert all_extracted_annotations
    return itertools.chain(
        vg_coco_gqa_grouped_metadata, epic_kitchens_grouped_metadata, alfred_grouped_metadata
    )
