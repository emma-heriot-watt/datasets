import itertools
from typing import Any

from rich.progress import Progress

from src.datamodels.constants import DatasetName
from src.parsers.align_multiple_datasets import AlignMultipleDatasets
from src.parsers.dataset_aligner import DatasetAligner


def test_dataset_aligner_works(
    dataset_aligner: DatasetAligner[Any, Any], progress: Progress
) -> None:
    aligned_metadata = dataset_aligner.get_aligned_metadata()

    source_metadata = [
        dataset_aligner.source_metadata_parser.convert_to_dataset_metadata(metadata)
        for metadata in list(dataset_aligner.source_metadata_parser.get_metadata(progress))
    ]
    target_metadata = [
        dataset_aligner.target_metadata_parser.convert_to_dataset_metadata(metadata)
        for metadata in list(dataset_aligner.target_metadata_parser.get_metadata(progress))
    ]

    input_metadata = list(itertools.chain(source_metadata, target_metadata))

    # Verify no metadata is lost during the process by asserting that each given is somewhere
    # in the output
    for metadata_group in itertools.chain.from_iterable(aligned_metadata):
        for metadata in metadata_group.values():
            assert metadata in input_metadata


def test_multiple_dataset_aligner_works(vg_coco_aligner, gqa_vg_aligner, progress):
    align_multiple_datasets = AlignMultipleDatasets(DatasetName.visual_genome, progress)

    aligned_metadata_iterable = [
        dataset_aligner.get_aligned_metadata()
        for dataset_aligner in (vg_coco_aligner, gqa_vg_aligner)
    ]

    all_metadata_groups = list(
        itertools.chain(*align_multiple_datasets(*aligned_metadata_iterable))
    )

    flattened_aligned_metadata = itertools.chain.from_iterable(
        itertools.chain(*aligned_metadata_iterable)
    )

    # Verify no metadata is lost during the process by asserting that each given is somewhere
    # in the output
    for metadata_groups in flattened_aligned_metadata:
        for metadata in metadata_groups.values():
            assert metadata in all_metadata_groups
