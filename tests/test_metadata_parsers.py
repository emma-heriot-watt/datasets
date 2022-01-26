from multiprocessing.pool import Pool
from typing import Any, Iterator

from pytest_cases import fixture, fixture_ref, parametrize
from rich.progress import Progress

from src.datamodels import DatasetMetadata
from src.parsers.dataset_metadata.metadata_parser import DatasetMetadataParser


@fixture
@parametrize(
    "metadata_parser",
    [
        fixture_ref("coco_metadata_parser"),
        fixture_ref("vg_metadata_parser"),
        fixture_ref("gqa_metadata_parser"),
        fixture_ref("epic_kitchens_metadata_parser"),
        fixture_ref("alfred_metadata_parser"),
    ],
)
def metadata_parser(metadata_parser):
    return metadata_parser


def test_metadata_parser_works(metadata_parser, progress):
    metadata = list(metadata_parser.get_metadata(progress))

    assert metadata

    for instance in metadata:
        assert isinstance(instance, metadata_parser.metadata_model)


def test_metadata_parser_works_with_multiprocessing(metadata_parser, progress):
    with Pool(2) as pool:
        metadata = list(metadata_parser.get_metadata(progress, pool))

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
