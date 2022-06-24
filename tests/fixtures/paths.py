from pathlib import Path
from typing import Any, Optional

from pytest_cases import fixture

from emma_datasets.io.paths import get_paths_from_dir


@fixture(scope="session")
def project_root() -> Path:
    return Path.joinpath(Path(__file__).parent.parent, "..").resolve()


@fixture(scope="session")
def fixtures_root(project_root: Path) -> Path:
    return Path.joinpath(project_root, "storage", "fixtures")


@fixture(scope="session")
def coco_captions_path(fixtures_root: Path) -> Path:
    paths = fixtures_root.joinpath("coco_captions.json")
    assert paths.exists()
    return paths


@fixture(scope="session")
def vg_image_data_path(fixtures_root: Path) -> Path:
    paths = fixtures_root.joinpath("vg_image_data.json")
    assert paths.exists()
    return paths


@fixture(scope="session")
def vg_regions_path(fixtures_root: Path) -> Path:
    paths = fixtures_root.joinpath("vg_regions.json")
    assert paths.exists()
    return paths


@fixture(scope="session")
def gqa_scene_graphs_path(fixtures_root: Path) -> Path:
    paths = fixtures_root.joinpath("gqa_scene_graph.json")
    assert paths.exists()
    return paths


@fixture(scope="session")
def gqa_questions_path(fixtures_root: Path) -> Path:
    paths = fixtures_root.joinpath("gqa_questions.json")
    assert paths.exists()
    return paths


@fixture(scope="session")
def ek_data_path(fixtures_root: Path) -> Path:
    paths = fixtures_root.joinpath("epic_kitchens.csv")
    assert paths.exists()
    return paths


@fixture(scope="session")
def ek_video_info_path(fixtures_root: Path) -> Path:
    paths = fixtures_root.joinpath("EPIC_100_video_info.csv")
    assert paths.exists()
    return paths


@fixture(scope="session")
def alfred_train_data_path(fixtures_root: Path) -> list[Path]:
    paths = list(get_paths_from_dir(fixtures_root.joinpath("alfred/train/")))
    for path in paths:
        assert path.exists()
    return paths


@fixture(scope="session")
def alfred_valid_seen_data_path(fixtures_root: Path) -> list[Path]:
    paths = list(get_paths_from_dir(fixtures_root.joinpath("alfred/valid_seen/")))
    for path in paths:
        assert path.exists()
    return paths


@fixture(scope="session")
def teach_edh_instance_path(fixtures_root: Path) -> Path:
    path = fixtures_root.joinpath("teach_edh")
    assert path.is_dir()
    return path


@fixture(scope="session")
def teach_edh_train_data_paths(teach_edh_instance_path: Path) -> list[Path]:
    root = teach_edh_instance_path.joinpath("train")
    return list(root.iterdir())


@fixture(scope="session")
def teach_edh_valid_seen_data_paths(teach_edh_instance_path: Path) -> list[Path]:
    root = teach_edh_instance_path.joinpath("valid_seen")
    return list(root.iterdir())


@fixture(scope="session")
def teach_edh_valid_unseen_data_paths(teach_edh_instance_path: Path) -> list[Path]:
    root = teach_edh_instance_path.joinpath("valid_unseen")
    return list(root.iterdir())


@fixture(scope="session")
def nlvr_instances_path(fixtures_root: Path) -> Path:
    return fixtures_root.joinpath("nlvr.jsonl")


@fixture(scope="session")
def teach_edh_all_data_paths(
    teach_edh_train_data_paths: list[Path],
    teach_edh_valid_seen_data_paths: list[Path],
    teach_edh_valid_unseen_data_paths: list[Path],
) -> list[Path]:
    return (
        teach_edh_train_data_paths
        + teach_edh_valid_seen_data_paths
        + teach_edh_valid_unseen_data_paths
    )


@fixture(scope="session")
def coco_instances_path(fixtures_root: Path) -> Path:
    return fixtures_root.joinpath("coco", "coco_captions_tiny.json")


@fixture(scope="session")
def vqa_v2_instance_path(fixtures_root: Path) -> Path:
    path = fixtures_root.joinpath("vqa_v2")
    assert path.is_dir()
    return path


@fixture(scope="session")
def vqa_v2_train_data_path(fixtures_root: Path) -> tuple[Path, Path]:
    split_path = fixtures_root.joinpath("vqa_v2/")
    questions_paths = split_path.joinpath("v2_OpenEnded_mscoco_train2014_questions.json")
    annotations_paths = split_path.joinpath("v2_mscoco_train2014_annotations.json")
    return (questions_paths, annotations_paths)


@fixture(scope="session")
def vqa_v2_valid_data_path(fixtures_root: Path) -> tuple[Path, Path]:
    split_path = fixtures_root.joinpath("vqa_v2/")

    questions_paths = split_path.joinpath("v2_OpenEnded_mscoco_val2014_questions.json")
    annotations_paths = split_path.joinpath("v2_mscoco_val2014_annotations.json")
    return (questions_paths, annotations_paths)


@fixture(scope="session")
def vqa_v2_test_data_path(fixtures_root: Path) -> tuple[Path, None]:
    questions_paths = fixtures_root.joinpath(
        "vqa_v2/v2_OpenEnded_mscoco_test-dev2015_questions.json"
    )
    return (questions_paths, None)


@fixture(scope="session")
def vqa_v2_all_data_paths(
    vqa_v2_train_data_path: tuple[Path, Path],
    vqa_v2_valid_data_path: tuple[Path, Path],
    vqa_v2_test_data_path: tuple[Path, None],
) -> list[tuple[Path, Optional[Path]]]:
    return [
        vqa_v2_train_data_path,
        vqa_v2_valid_data_path,
        vqa_v2_test_data_path,
    ]


@fixture(scope="session")
def extracted_annotations_paths(request: Any) -> dict[str, Path]:
    """Create cached folders for the extracted annotations.

    To create a separate folder to cache the annotations, just add a new entry to the
    `annotation_folders` list.
    """
    annotation_folders = [
        "ek_captions",
        "coco_captions",
        "alfred_captions",
        "scene_graphs",
        "trajectories",
        "regions",
        "qa_pairs",
        "task_descriptions",
    ]

    paths = {}

    for name in annotation_folders:
        paths[name] = Path(request.config.cache.makedir(name))

    return paths


@fixture(scope="session")
def cached_db_dir_path(request: Any) -> Path:
    return Path(request.config.cache.makedir("db"))


@fixture(scope="session")
def cached_instances_db_path(cached_db_dir_path: Path) -> Path:
    return cached_db_dir_path.joinpath("instances.db")


@fixture(scope="session")
def pretrain_instances_db_path(fixtures_root: Path) -> Path:
    path = Path(fixtures_root.joinpath("db", "instances.db"))
    assert path.exists()
    return path
