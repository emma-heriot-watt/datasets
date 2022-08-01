import itertools
from pathlib import Path

from pydantic import parse_file_as

from emma_datasets.datamodels import (
    ActionTrajectory,
    Caption,
    QuestionAnswerPair,
    Region,
    SceneGraph,
)


def test_coco_caption_extractor_works(
    extract_coco_captions: bool, extracted_annotations_paths: dict[str, Path]
) -> None:
    assert extract_coco_captions

    generated_caption_files = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path)
            for file_path in extracted_annotations_paths["coco_captions"].iterdir()
        )
    )

    assert generated_caption_files


def test_vg_region_extractor_works(
    extract_vg_regions: bool, extracted_annotations_paths: dict[str, Path]
) -> None:
    assert extract_vg_regions

    generated_region_files = [
        parse_file_as(list[Region], file_path)
        for file_path in extracted_annotations_paths["regions"].iterdir()
    ]

    assert generated_region_files


def test_gqa_qa_extractor_works(
    extract_gqa_qa_pairs: bool, extracted_annotations_paths: dict[str, Path]
) -> None:
    assert extract_gqa_qa_pairs

    generated_qa_pairs = [
        parse_file_as(list[QuestionAnswerPair], file_path)
        for file_path in extracted_annotations_paths["qa_pairs"].iterdir()
    ]
    assert generated_qa_pairs


def test_vqa_v2_qa_extractor_works(
    extract_vqa_v2_qa_pairs: bool, extracted_annotations_paths: dict[str, Path]
) -> None:
    assert extract_vqa_v2_qa_pairs

    generated_qa_pairs = [
        parse_file_as(list[QuestionAnswerPair], file_path)
        for file_path in extracted_annotations_paths["qa_pairs"].iterdir()
    ]
    assert generated_qa_pairs


def test_gqa_scene_graph_extractor_works(
    extract_gqa_scene_graphs: bool, extracted_annotations_paths: dict[str, Path]
) -> None:
    assert extract_gqa_scene_graphs

    generated_scene_graphs = [
        parse_file_as(SceneGraph, file_path)
        for file_path in extracted_annotations_paths["scene_graphs"].iterdir()
    ]

    assert generated_scene_graphs


def test_epic_kitchens_caption_extractor_works(
    extract_epic_kitchen_captions: bool, extracted_annotations_paths: dict[str, Path]
) -> None:
    assert extract_epic_kitchen_captions

    generated_captions = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path)
            for file_path in extracted_annotations_paths["ek_captions"].iterdir()
        )
    )

    assert generated_captions


def test_alfred_caption_extractor_works(
    extract_alfred_captions: bool, extracted_annotations_paths: dict[str, Path]
) -> None:
    assert extract_alfred_captions

    generated_captions = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path)
            for file_path in extracted_annotations_paths["alfred_captions"].iterdir()
        )
    )

    assert generated_captions


def test_alfred_subgoal_trajectory_extractor_works(
    extract_alfred_subgoal_trajectories: bool, extracted_annotations_paths: dict[str, Path]
) -> None:
    assert extract_alfred_subgoal_trajectories

    generated_trajectories = [
        parse_file_as(ActionTrajectory, file_path)
        for file_path in extracted_annotations_paths["trajectories"].iterdir()
    ]

    assert generated_trajectories
