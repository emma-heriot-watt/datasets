import itertools
from pathlib import Path
from typing import Any

from pydantic import parse_file_as
from pytest_cases import parametrize_with_cases
from rich.progress import Progress

from src.datamodels import ActionTrajectory, Caption, QuestionAnswerPair, Region, SceneGraph
from src.parsers.instance_splitters import (
    AlfredCaptionSplitter,
    CocoCaptionSplitter,
    EpicKitchensCaptionSplitter,
    GqaQaPairSplitter,
    GqaSceneGraphSplitter,
    VgRegionsSplitter,
)
from src.parsers.instance_splitters.alfred_trajectories import AlfredTrajectorySplitter


def split_instances(
    instance_splitter_class: Any,
    paths: list[Path],
    tmp_path: Path,
) -> None:
    with Progress() as progress:
        instance_splitter = instance_splitter_class(
            paths=paths,
            output_dir=tmp_path,
            progress=progress,
        )
        instance_splitter.run(progress)


@parametrize_with_cases("paths", cases=".conftest")
def test_coco_caption_splitter_works(paths: dict[str, Path], tmp_path: Path) -> None:
    split_instances(
        CocoCaptionSplitter, [paths["coco_caption_train"], paths["coco_caption_val"]], tmp_path
    )

    generated_caption_files = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path) for file_path in tmp_path.iterdir()
        )
    )

    assert generated_caption_files


@parametrize_with_cases("paths", cases=".conftest")
def test_vg_region_splitter_works(paths: dict[str, Path], tmp_path: Path) -> None:
    split_instances(VgRegionsSplitter, [paths["vg_regions"]], tmp_path)

    generated_region_files = [
        parse_file_as(list[Region], file_path) for file_path in tmp_path.iterdir()
    ]

    assert generated_region_files


@parametrize_with_cases("paths", cases=".conftest")
def test_gqa_qa_splitter_works(paths: dict[str, list[Path]], tmp_path: Path) -> None:
    split_instances(GqaQaPairSplitter, paths["gqa_questions"], tmp_path)

    generated_qa_pairs = [
        parse_file_as(list[QuestionAnswerPair], file_path) for file_path in tmp_path.iterdir()
    ]
    assert generated_qa_pairs


@parametrize_with_cases("paths", cases=".conftest")
def test_gqa_scene_graph_splitter_works(paths: dict[str, Path], tmp_path: Path) -> None:
    split_instances(
        GqaSceneGraphSplitter,
        [paths["gqa_scene_graph_train"], paths["gqa_scene_graph_val"]],
        tmp_path,
    )

    generated_scene_graphs = [
        parse_file_as(SceneGraph, file_path) for file_path in tmp_path.iterdir()
    ]

    assert generated_scene_graphs


@parametrize_with_cases("paths", cases=".conftest")
def test_epic_kitchens_caption_splitter_works(paths: dict[str, Path], tmp_path: Path) -> None:
    split_instances(EpicKitchensCaptionSplitter, [paths["ek_train"], paths["ek_val"]], tmp_path)

    generated_captions = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path) for file_path in tmp_path.iterdir()
        )
    )

    assert generated_captions


@parametrize_with_cases("paths", cases=".conftest")
def test_alfred_caption_splitter_works(paths: dict[str, list[Path]], tmp_path: Path) -> None:
    instance_paths = list(itertools.chain(paths["alfred_train"], paths["alfred_valid_seen"]))

    split_instances(AlfredCaptionSplitter, instance_paths, tmp_path)

    generated_captions = list(
        itertools.chain.from_iterable(
            parse_file_as(list[Caption], file_path) for file_path in tmp_path.iterdir()
        )
    )

    assert generated_captions


@parametrize_with_cases("paths", cases=".conftest")
def test_alfred_trajectory_splitter_works(paths: dict[str, list[Path]], tmp_path: Path) -> None:
    instance_paths = list(itertools.chain(paths["alfred_train"], paths["alfred_valid_seen"]))

    split_instances(AlfredTrajectorySplitter, instance_paths, tmp_path)

    generated_trajectories = [
        parse_file_as(ActionTrajectory, file_path) for file_path in tmp_path.iterdir()
    ]

    assert generated_trajectories
