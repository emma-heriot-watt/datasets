from multiprocessing.pool import Pool

from pytest_cases import fixture, fixture_ref, parametrize

from src.datamodels import DatasetMetadata


@fixture
@parametrize(
    "metadata_parser",
    [
        fixture_ref("coco_metadata_parser"),
        fixture_ref("vg_metadata_parser"),
        fixture_ref("gqa_metadata_parser"),
        fixture_ref("epic_kitchens_metadata_parser"),
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


def test_metadata_parser_can_convert_to_dataset_metadata(metadata_parser, progress):
    structured_metadata = list(metadata_parser.get_metadata(progress))

    dataset_metadata = [
        metadata_parser.convert_to_dataset_metadata(metadata) for metadata in structured_metadata
    ]

    for instance in dataset_metadata:
        assert isinstance(instance, DatasetMetadata)
