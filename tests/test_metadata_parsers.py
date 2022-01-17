from multiprocessing.pool import Pool
from pathlib import Path

import pytest
from rich.progress import Progress

from src.datamodels import DatasetMetadata
from src.parsers.dataset_metadata import CocoMetadataParser, GqaMetadataParser, VgMetadataParser


@pytest.fixture
def progress() -> Progress:
    return Progress()


def coco_parser(fixtures_path, progress):
    fake_caption_dir = Path("caption")
    fake_image_dir = Path("images")

    return CocoMetadataParser(
        caption_train_path=fixtures_path.joinpath("coco_captions.json"),
        caption_val_path=fixtures_path.joinpath("coco_captions.json"),
        images_dir=fake_image_dir,
        captions_dir=fake_caption_dir,
        progress=progress,
    )


def vg_parser(fixtures_path, progress):
    fake_images_dir = Path("images")
    fake_regions_dir = Path("regions")

    return VgMetadataParser(
        image_data_json_path=fixtures_path.joinpath("vg_image_data.json"),
        images_dir=fake_images_dir,
        regions_dir=fake_regions_dir,
        progress=progress,
    )


def gqa_parser(fixtures_path, progress):
    return GqaMetadataParser(
        scene_graphs_train_path=fixtures_path.joinpath("gqa_scene_graph.json"),
        scene_graphs_val_path=fixtures_path.joinpath("gqa_scene_graph.json"),
        images_dir=Path("images"),
        scene_graphs_dir=Path("scene_graphs"),
        qa_pairs_dir=Path("qa_pairs"),
        progress=progress,
    )


@pytest.fixture
def parser(request, fixtures_path, progress):
    parser_type_switcher = {
        "COCO": coco_parser(fixtures_path, progress),
        "Visual Genome": vg_parser(fixtures_path, progress),
        "GQA": gqa_parser(fixtures_path, progress),
    }

    return parser_type_switcher[request.param]


@pytest.mark.parametrize("parser", ["COCO", "Visual Genome", "GQA"], indirect=True)
class TestMetadataParser:
    def test_get_metadata(self, parser, progress):
        metadata = list(parser.get_metadata(progress))

        assert metadata

        for instance in metadata:
            assert isinstance(instance, parser.metadata_model)

    def test_get_metadata_with_pool(self, parser, progress):
        with Pool(2) as pool:
            metadata = list(parser.get_metadata(progress, pool))

        assert metadata

        for instance in metadata:
            assert isinstance(instance, parser.metadata_model)

    def test_convert_to_dataset_metadata(self, parser, progress):
        structured_metadata = list(parser.get_metadata(progress))

        dataset_metadata = [
            parser.convert_to_dataset_metadata(metadata) for metadata in structured_metadata
        ]

        for instance in dataset_metadata:
            assert isinstance(instance, DatasetMetadata)
