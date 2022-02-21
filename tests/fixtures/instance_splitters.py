import itertools
from pathlib import Path
from typing import Any

from pytest_cases import fixture

from emma_datasets.common import get_progress
from emma_datasets.parsers.instance_splitters import (
    AlfredCaptionSplitter,
    AlfredTrajectorySplitter,
    CocoCaptionSplitter,
    EpicKitchensCaptionSplitter,
    GqaQaPairSplitter,
    GqaSceneGraphSplitter,
    VgRegionsSplitter,
)


def split_instances(
    instance_splitter_class: Any,
    paths: list[Path],
    output_dir: Path,
) -> None:
    with get_progress() as progress:
        instance_splitter = instance_splitter_class(
            paths=paths,
            output_dir=output_dir,
            progress=progress,
        )
        instance_splitter.run(progress)


@fixture
def split_coco_captions(coco_captions_path: Path, split_instances_paths: dict[str, Path]) -> bool:
    split_instances(
        CocoCaptionSplitter,
        [coco_captions_path, coco_captions_path],
        split_instances_paths["coco_captions"],
    )
    return True


@fixture
def split_vg_regions(vg_regions_path: Path, split_instances_paths: dict[str, Path]) -> bool:
    split_instances(VgRegionsSplitter, [vg_regions_path], split_instances_paths["regions"])
    return True


@fixture
def split_gqa_qa_pairs(gqa_questions_path: Path, split_instances_paths: dict[str, Path]) -> bool:
    split_instances(GqaQaPairSplitter, [gqa_questions_path], split_instances_paths["qa_pairs"])
    return True


@fixture
def split_gqa_scene_graphs(
    gqa_scene_graphs_path: Path, split_instances_paths: dict[str, Path]
) -> bool:
    split_instances(
        GqaSceneGraphSplitter,
        [gqa_scene_graphs_path, gqa_scene_graphs_path],
        split_instances_paths["scene_graphs"],
    )
    return True


@fixture
def split_epic_kitchen_captions(
    ek_data_path: Path, split_instances_paths: dict[str, Path]
) -> bool:
    split_instances(
        EpicKitchensCaptionSplitter,
        [ek_data_path, ek_data_path],
        split_instances_paths["ek_captions"],
    )
    return True


@fixture
def split_alfred_captions(
    alfred_train_data_path: list[Path],
    alfred_valid_seen_data_path: list[Path],
    split_instances_paths: dict[str, Path],
) -> bool:
    instance_paths = list(itertools.chain(alfred_train_data_path, alfred_valid_seen_data_path))

    split_instances(
        AlfredCaptionSplitter, instance_paths, split_instances_paths["alfred_captions"]
    )
    return True


@fixture
def split_alfred_trajectories(
    alfred_train_data_path: list[Path],
    alfred_valid_seen_data_path: list[Path],
    split_instances_paths: dict[str, Path],
) -> bool:
    instance_paths = list(itertools.chain(alfred_train_data_path, alfred_valid_seen_data_path))

    split_instances(
        AlfredTrajectorySplitter, instance_paths, split_instances_paths["trajectories"]
    )
    return True


@fixture
def all_split_instances(
    split_coco_captions: bool,
    split_vg_regions: bool,
    split_gqa_qa_pairs: bool,
    split_gqa_scene_graphs: bool,
    split_epic_kitchen_captions: bool,
    split_alfred_captions: bool,
    split_alfred_trajectories: bool,
) -> bool:
    assert split_coco_captions
    assert split_vg_regions
    assert split_gqa_qa_pairs
    assert split_gqa_scene_graphs
    assert split_epic_kitchen_captions
    assert split_alfred_captions
    assert split_alfred_trajectories

    return True
