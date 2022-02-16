import itertools
from pathlib import Path

from pydantic import parse_file_as

from emma_datasets.datamodels import (
    Caption,
    GenericActionTrajectory,
    QuestionAnswerPair,
    Region,
    SceneGraph,
)
from emma_datasets.datamodels.datasets.alfred import AlfredHighAction, AlfredLowAction


def test_coco_caption_splitter_works(
    split_coco_captions: bool, split_instances_paths: dict[str, Path]
) -> None:
    assert split_coco_captions

    generated_caption_files = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path)
            for file_path in split_instances_paths["coco_captions"].iterdir()
        )
    )

    assert generated_caption_files


def test_vg_region_splitter_works(
    split_vg_regions: bool, split_instances_paths: dict[str, Path]
) -> None:
    assert split_vg_regions

    generated_region_files = [
        parse_file_as(list[Region], file_path)
        for file_path in split_instances_paths["regions"].iterdir()
    ]

    assert generated_region_files


def test_gqa_qa_splitter_works(
    split_gqa_qa_pairs: bool, split_instances_paths: dict[str, Path]
) -> None:
    assert split_gqa_qa_pairs

    generated_qa_pairs = [
        parse_file_as(list[QuestionAnswerPair], file_path)
        for file_path in split_instances_paths["qa_pairs"].iterdir()
    ]
    assert generated_qa_pairs


def test_gqa_scene_graph_splitter_works(
    split_gqa_scene_graphs: bool, split_instances_paths: dict[str, Path]
) -> None:
    assert split_gqa_scene_graphs

    generated_scene_graphs = [
        parse_file_as(SceneGraph, file_path)
        for file_path in split_instances_paths["scene_graphs"].iterdir()
    ]

    assert generated_scene_graphs


def test_epic_kitchens_caption_splitter_works(
    split_epic_kitchen_captions: bool, split_instances_paths: dict[str, Path]
) -> None:
    assert split_epic_kitchen_captions

    generated_captions = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path)
            for file_path in split_instances_paths["ek_captions"].iterdir()
        )
    )

    assert generated_captions


def test_alfred_caption_splitter_works(
    split_alfred_captions: bool, split_instances_paths: dict[str, Path]
) -> None:
    assert split_alfred_captions

    generated_captions = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path)
            for file_path in split_instances_paths["alfred_captions"].iterdir()
        )
    )

    assert generated_captions


def test_alfred_trajectory_splitter_works(
    split_alfred_trajectories: bool, split_instances_paths: dict[str, Path]
) -> None:
    assert split_alfred_trajectories

    generated_trajectories = [
        parse_file_as(GenericActionTrajectory[AlfredLowAction, AlfredHighAction], file_path)
        for file_path in split_instances_paths["trajectories"].iterdir()
    ]

    assert generated_trajectories
