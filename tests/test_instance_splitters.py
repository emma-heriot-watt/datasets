import itertools

# import pytest
from pydantic import parse_file_as
from rich.progress import Progress

from src.datamodels import Caption, QuestionAnswerPair, Region, SceneGraph
from src.parsers.instance_splitters import (
    CocoCaptionSplitter,
    GqaQaPairSplitter,
    GqaSceneGraphSplitter,
    VgRegionsSplitter,
)


def split_instances(instance_splitter_class, source_file, tmp_path, fixtures_path):
    with Progress() as progress:
        instance_splitter = instance_splitter_class(
            paths=fixtures_path.joinpath(source_file),
            output_dir=tmp_path,
            progress=progress,
        )
        instance_splitter.run(progress)


def test_coco_caption_splitter(tmp_path, fixtures_path):
    total_captions_count = 14

    split_instances(CocoCaptionSplitter, "coco_captions.json", tmp_path, fixtures_path)

    generated_caption_files = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path) for file_path in tmp_path.iterdir()
        )
    )

    assert len(generated_caption_files) == total_captions_count


def test_vg_region_splitter(tmp_path, fixtures_path):
    total_region_count = 5

    split_instances(VgRegionsSplitter, "region_descriptions.json", tmp_path, fixtures_path)

    generated_region_files = [
        parse_file_as(list[Region], file_path) for file_path in tmp_path.iterdir()
    ]

    assert len(generated_region_files) == total_region_count


def test_gqa_qa_pairs_splitter(tmp_path, fixtures_path):
    split_instances(GqaQaPairSplitter, "gqa_questions.json", tmp_path, fixtures_path)

    generated_qa_pairs = [
        parse_file_as(list[QuestionAnswerPair], file_path) for file_path in tmp_path.iterdir()
    ]

    unique_image_ids = 4
    total_qa_pairs = 5

    assert len(generated_qa_pairs) == unique_image_ids
    assert len([qa_pair for image in generated_qa_pairs for qa_pair in image]) == total_qa_pairs


def test_gqa_scene_graph_splitter(tmp_path, fixtures_path):
    split_instances(GqaSceneGraphSplitter, "gqa_scene_graph.json", tmp_path, fixtures_path)

    generated_scene_graphs = [
        parse_file_as(SceneGraph, file_path) for file_path in tmp_path.iterdir()
    ]

    assert len(generated_scene_graphs) == 3
