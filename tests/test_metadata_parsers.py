from typing import Any, Iterator

from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata
from emma_datasets.parsers.dataset_metadata.metadata_parser import DatasetMetadataParser


def test_metadata_parser_works(metadata_parser, progress):
    metadata = list(metadata_parser.get_metadata(progress))

    assert metadata

    for instance in metadata:
        assert isinstance(instance, metadata_parser.metadata_model)


def test_metadata_parser_can_convert_to_dataset_metadata(
    metadata_parser: DatasetMetadataParser[Any], progress: Progress
) -> None:
    structured_metadata = list(metadata_parser.get_metadata(progress))

    dataset_metadata = (
        metadata_parser.convert_to_dataset_metadata(metadata) for metadata in structured_metadata
    )

    for instance in dataset_metadata:
        if isinstance(instance, Iterator):
            for yielded_instance in instance:
                assert isinstance(yielded_instance, DatasetMetadata)
        else:
            assert isinstance(instance, DatasetMetadata)
