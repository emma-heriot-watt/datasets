from pathlib import Path
from typing import Any

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
def teach_edh_train_data_paths(fixtures_root: Path) -> list[Path]:
    train_data_root = fixtures_root.joinpath("teach_edh", "train")
    return list(train_data_root.iterdir())


@fixture(scope="session")
def teach_edh_valid_seen_data_paths(fixtures_root: Path) -> list[Path]:
    root = fixtures_root.joinpath("teach_edh", "valid_seen")
    return list(root.iterdir())


@fixture(scope="session")
def teach_edh_valid_unseen_data_paths(fixtures_root: Path) -> list[Path]:
    root = fixtures_root.joinpath("teach_edh", "valid_unseen")
    return list(root.iterdir())


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
def split_instances_paths(request: Any) -> dict[str, Path]:
    annotation_folders = [
        "ek_captions",
        "coco_captions",
        "alfred_captions",
        "scene_graphs",
        "trajectories",
        "regions",
        "qa_pairs",
    ]

    split_instances_paths = {}

    for name in annotation_folders:
        split_instances_paths[name] = Path(request.config.cache.makedir(name))

    return split_instances_paths


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
