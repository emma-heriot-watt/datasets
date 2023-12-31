import itertools
from pathlib import Path
from typing import Any, Union

from pytest_cases import fixture

from emma_datasets.common import get_progress
from emma_datasets.parsers.annotation_extractors import (
    AlfredCaptionExtractor,
    AlfredTaskDescriptionExtractor,
    AlfredTrajectoryExtractor,
    CocoCaptionExtractor,
    EpicKitchensCaptionExtractor,
    GqaQaPairExtractor,
    GqaSceneGraphExtractor,
    VgRegionsExtractor,
    VQAv2QaPairExtractor,
)


def extracted_annotations(
    annotation_extractor_class: Any,
    paths: Union[list[Path], list[tuple[Path, Path]]],
    output_dir: Path,
) -> None:
    with get_progress() as progress:
        annotation_extractor = annotation_extractor_class(
            paths=paths,
            output_dir=output_dir,
            progress=progress,
        )
        annotation_extractor.run(progress)


@fixture
def extract_coco_captions(
    coco_captions_path_train: Path,
    coco_captions_path_valid: Path,
    extracted_annotations_paths: dict[str, Path],
) -> bool:
    extracted_annotations(
        CocoCaptionExtractor,
        [coco_captions_path_train, coco_captions_path_valid],
        extracted_annotations_paths["coco_captions"],
    )
    return True


@fixture
def extract_vg_regions(
    vg_regions_path: Path, extracted_annotations_paths: dict[str, Path]
) -> bool:
    extracted_annotations(
        VgRegionsExtractor, [vg_regions_path], extracted_annotations_paths["regions"]
    )
    return True


@fixture
def extract_gqa_qa_pairs(
    gqa_questions_path: Path, extracted_annotations_paths: dict[str, Path]
) -> bool:
    extracted_annotations(
        GqaQaPairExtractor, [gqa_questions_path], extracted_annotations_paths["qa_pairs"]
    )
    return True


@fixture
def extract_vqa_v2_qa_pairs(
    vqa_v2_train_data_path: tuple[Path, Path],
    vqa_v2_valid_data_path: tuple[Path, Path],
    extracted_annotations_paths: dict[str, Path],
) -> bool:
    extracted_annotations(
        VQAv2QaPairExtractor,
        [vqa_v2_train_data_path, vqa_v2_valid_data_path],
        extracted_annotations_paths["qa_pairs"],
    )
    return True


@fixture
def extract_gqa_scene_graphs(
    gqa_scene_graphs_path: Path, extracted_annotations_paths: dict[str, Path]
) -> bool:
    extracted_annotations(
        GqaSceneGraphExtractor,
        [gqa_scene_graphs_path, gqa_scene_graphs_path],
        extracted_annotations_paths["scene_graphs"],
    )
    return True


@fixture
def extract_epic_kitchen_captions(
    ek_data_path: Path, extracted_annotations_paths: dict[str, Path]
) -> bool:
    extracted_annotations(
        EpicKitchensCaptionExtractor,
        [ek_data_path, ek_data_path],
        extracted_annotations_paths["ek_captions"],
    )
    return True


@fixture
def extract_alfred_captions(
    alfred_train_data_path: list[Path],
    alfred_valid_seen_data_path: list[Path],
    extracted_annotations_paths: dict[str, Path],
) -> bool:
    instance_paths = list(itertools.chain(alfred_train_data_path, alfred_valid_seen_data_path))

    extracted_annotations(
        AlfredCaptionExtractor, instance_paths, extracted_annotations_paths["alfred_captions"]
    )
    return True


@fixture
def extract_alfred_task_descriptions(
    alfred_train_data_path: list[Path],
    alfred_valid_seen_data_path: list[Path],
    extracted_annotations_paths: dict[str, Path],
) -> bool:
    instance_paths = list(itertools.chain(alfred_train_data_path, alfred_valid_seen_data_path))

    extracted_annotations(
        AlfredTaskDescriptionExtractor,
        instance_paths,
        extracted_annotations_paths["task_descriptions"],
    )
    return True


@fixture
def extract_alfred_subgoal_trajectories(
    alfred_train_data_path: list[Path],
    alfred_valid_seen_data_path: list[Path],
    extracted_annotations_paths: dict[str, Path],
) -> bool:
    instance_paths = list(itertools.chain(alfred_train_data_path, alfred_valid_seen_data_path))

    extracted_annotations(
        AlfredTrajectoryExtractor,
        instance_paths,
        extracted_annotations_paths["trajectories"],
    )
    return True


@fixture
def all_extracted_annotations(
    extract_coco_captions: bool,
    extract_vg_regions: bool,
    extract_gqa_qa_pairs: bool,
    extract_vqa_v2_qa_pairs: bool,
    extract_gqa_scene_graphs: bool,
    extract_epic_kitchen_captions: bool,
    extract_alfred_captions: bool,
    extract_alfred_task_descriptions: bool,
    extract_alfred_subgoal_trajectories: bool,
) -> bool:
    assert extract_coco_captions
    assert extract_vg_regions
    assert extract_gqa_qa_pairs
    assert extract_vqa_v2_qa_pairs
    assert extract_gqa_scene_graphs
    assert extract_epic_kitchen_captions
    assert extract_alfred_captions
    assert extract_alfred_task_descriptions
    assert extract_alfred_subgoal_trajectories

    return True
