import itertools
import os
from pathlib import Path
from typing import Any, Iterator, Union, cast

import pytest
from pytest_cases import fixture, parametrize
from rich.progress import Progress

from emma_datasets.common import get_progress
from emma_datasets.common.settings import Settings
from emma_datasets.datamodels import DatasetMetadata, DatasetName, DatasetSplit
from emma_datasets.datamodels.datasets import CocoImageMetadata, GqaImageMetadata, VgImageMetadata
from emma_datasets.db.dataset_db import DatasetDb
from emma_datasets.io.paths import get_paths_from_dir
from emma_datasets.parsers.align_multiple_datasets import AlignMultipleDatasets
from emma_datasets.parsers.dataset_aligner import DatasetAligner
from emma_datasets.parsers.dataset_metadata import (
    AlfredMetadataParser,
    CocoMetadataParser,
    DataPathTuple,
    EpicKitchensMetadataParser,
    GqaMetadataParser,
    VgMetadataParser,
)
from emma_datasets.parsers.instance_splitters import (
    AlfredCaptionSplitter,
    AlfredTrajectorySplitter,
    CocoCaptionSplitter,
    EpicKitchensCaptionSplitter,
    GqaQaPairSplitter,
    GqaSceneGraphSplitter,
    VgRegionsSplitter,
)


FIXTURES_PATH = Path("./storage/fixtures")
DATASETS_PATH = Path("./storage/datasets")


if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture
def progress() -> Progress:
    return get_progress()


# ----------------------------------- Cases ---------------------------------- #


@fixture(scope="session")
def paths() -> dict[str, Union[Path, list[Path]]]:
    return {
        "base": FIXTURES_PATH,
        "coco_caption_train": FIXTURES_PATH.joinpath("coco_captions.json"),
        "coco_caption_val": FIXTURES_PATH.joinpath("coco_captions.json"),
        "vg_image_data": FIXTURES_PATH.joinpath("vg_image_data.json"),
        "vg_regions": FIXTURES_PATH.joinpath("vg_regions.json"),
        "gqa_scene_graph_train": FIXTURES_PATH.joinpath("gqa_scene_graph.json"),
        "gqa_scene_graph_val": FIXTURES_PATH.joinpath("gqa_scene_graph.json"),
        "gqa_questions": [FIXTURES_PATH.joinpath("gqa_questions.json")],
        "ek_train": FIXTURES_PATH.joinpath("epic_kitchens.csv"),
        "ek_val": FIXTURES_PATH.joinpath("epic_kitchens.csv"),
        "ek_video_info": FIXTURES_PATH.joinpath("EPIC_100_video_info.csv"),
        "alfred_train": list(get_paths_from_dir(FIXTURES_PATH.joinpath("alfred/train/"))),
        "alfred_valid_seen": list(
            get_paths_from_dir(FIXTURES_PATH.joinpath("alfred/valid_seen/"))
        ),
    }


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


settings = Settings()


# ---------------------------- Instance splitters ---------------------------- #
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
def split_coco_captions(paths: dict[str, Path], split_instances_paths: dict[str, Path]) -> bool:
    split_instances(
        CocoCaptionSplitter,
        [paths["coco_caption_train"], paths["coco_caption_val"]],
        split_instances_paths["coco_captions"],
    )

    return True


@fixture
def split_vg_regions(paths: dict[str, Path], split_instances_paths: dict[str, Path]) -> bool:
    split_instances(VgRegionsSplitter, [paths["vg_regions"]], split_instances_paths["regions"])
    return True


@fixture
def split_gqa_qa_pairs(
    paths: dict[str, list[Path]], split_instances_paths: dict[str, Path]
) -> bool:
    split_instances(GqaQaPairSplitter, paths["gqa_questions"], split_instances_paths["qa_pairs"])
    return True


@fixture
def split_gqa_scene_graphs(paths: dict[str, Path], split_instances_paths: dict[str, Path]) -> bool:
    split_instances(
        GqaSceneGraphSplitter,
        [paths["gqa_scene_graph_train"], paths["gqa_scene_graph_val"]],
        split_instances_paths["scene_graphs"],
    )
    return True


@fixture
def split_epic_kitchen_captions(
    paths: dict[str, Path], split_instances_paths: dict[str, Path]
) -> bool:
    split_instances(
        EpicKitchensCaptionSplitter,
        [paths["ek_train"], paths["ek_val"]],
        split_instances_paths["ek_captions"],
    )
    return True


@fixture
def split_alfred_captions(
    paths: dict[str, list[Path]], split_instances_paths: dict[str, Path]
) -> bool:
    instance_paths = list(itertools.chain(paths["alfred_train"], paths["alfred_valid_seen"]))

    split_instances(
        AlfredCaptionSplitter, instance_paths, split_instances_paths["alfred_captions"]
    )
    return True


@fixture
def split_alfred_trajectories(
    paths: dict[str, list[Path]], split_instances_paths: dict[str, Path]
) -> bool:
    instance_paths = list(itertools.chain(paths["alfred_train"], paths["alfred_valid_seen"]))

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


# ----------------------------- Metadata Parsers ----------------------------- #


@fixture
def coco_metadata_parser(
    paths: dict[str, Path], split_instances_paths: dict[str, Path], progress: Progress
) -> CocoMetadataParser:
    return CocoMetadataParser(
        caption_train_path=paths["coco_caption_train"],
        caption_val_path=paths["coco_caption_val"],
        images_dir=settings.paths.coco_images,
        captions_dir=split_instances_paths["coco_captions"],
        features_dir=settings.paths.coco_features,
        progress=progress,
    )


@fixture
def vg_metadata_parser(
    paths: dict[str, Path], split_instances_paths: dict[str, Path], progress: Progress
) -> VgMetadataParser:
    return VgMetadataParser(
        image_data_json_path=paths["vg_image_data"],
        images_dir=settings.paths.visual_genome_images,
        regions_dir=split_instances_paths["regions"],
        features_dir=settings.paths.visual_genome_features,
        progress=progress,
    )


@fixture
def gqa_metadata_parser(
    paths: dict[str, Path], split_instances_paths: dict[str, Path], progress: Progress
) -> GqaMetadataParser:
    return GqaMetadataParser(
        scene_graphs_train_path=paths["gqa_scene_graph_train"],
        scene_graphs_val_path=paths["gqa_scene_graph_val"],
        images_dir=settings.paths.gqa_images,
        scene_graphs_dir=split_instances_paths["scene_graphs"],
        qa_pairs_dir=split_instances_paths["qa_pairs"],
        features_dir=settings.paths.gqa_features,
        progress=progress,
    )


@fixture
def epic_kitchens_metadata_parser(
    paths: dict[str, Path], split_instances_paths: dict[str, Path], progress: Progress
) -> EpicKitchensMetadataParser:
    data_paths: list[DataPathTuple] = [
        (paths["ek_train"], DatasetSplit.train),
        (paths["ek_val"], DatasetSplit.valid),
    ]
    return EpicKitchensMetadataParser(
        data_paths=data_paths,
        frames_dir=settings.paths.epic_kitchens_frames,
        captions_dir=split_instances_paths["ek_captions"],
        features_dir=settings.paths.epic_kitchens_features,
        video_info_file=paths["ek_video_info"],
        progress=progress,
    )


@fixture
def alfred_metadata_parser(
    paths: dict[str, list[Path]], split_instances_paths: dict[str, Path], progress: Progress
) -> AlfredMetadataParser:
    data_paths: list[DataPathTuple] = list(
        itertools.chain.from_iterable(
            [
                zip(paths["alfred_train"], itertools.cycle([DatasetSplit.train])),
                zip(paths["alfred_valid_seen"], itertools.cycle([DatasetSplit.valid])),
            ]
        )
    )

    base_path: Path = cast(Path, paths["base"])

    return AlfredMetadataParser(
        data_paths=data_paths,
        alfred_dir=base_path.joinpath("alfred/"),
        captions_dir=split_instances_paths["alfred_captions"],
        trajectories_dir=split_instances_paths["trajectories"],
        features_dir=settings.paths.alfred_features,
        progress=progress,
    )


# ------------------------------ Dataset Aligner ----------------------------- #
@fixture
def vg_coco_aligner(
    vg_metadata_parser: VgMetadataParser,
    coco_metadata_parser: CocoMetadataParser,
    progress: Progress,
) -> DatasetAligner[VgImageMetadata, CocoImageMetadata]:
    return DatasetAligner[VgImageMetadata, CocoImageMetadata](
        vg_metadata_parser,
        coco_metadata_parser,
        source_mapping_attr_for_target="coco_id",
        target_mapping_attr_for_source="id",
        progress=progress,
    )


@fixture
def gqa_vg_aligner(
    vg_metadata_parser: VgMetadataParser,
    gqa_metadata_parser: GqaMetadataParser,
    progress: Progress,
) -> DatasetAligner[GqaImageMetadata, VgImageMetadata]:
    return DatasetAligner[GqaImageMetadata, VgImageMetadata](
        gqa_metadata_parser,
        vg_metadata_parser,
        source_mapping_attr_for_target="id",
        target_mapping_attr_for_source="image_id",
        progress=progress,
    )


@fixture
@parametrize("dataset_aligner", [vg_coco_aligner, gqa_vg_aligner])
def dataset_aligner(dataset_aligner):
    return dataset_aligner


# ----------------------------- Grouped metadata ----------------------------- #
@fixture
def vg_coco_gqa_grouped_metadata(
    vg_coco_aligner: DatasetAligner[VgImageMetadata, CocoImageMetadata],
    gqa_vg_aligner: DatasetAligner[GqaImageMetadata, VgImageMetadata],
    progress: Progress,
) -> Iterator[list[DatasetMetadata]]:
    align_multiple_datasets = AlignMultipleDatasets(DatasetName.visual_genome, progress)
    aligned_metadata_iterable = [
        dataset_aligner.get_aligned_metadata()
        for dataset_aligner in (vg_coco_aligner, gqa_vg_aligner)
    ]
    all_metadata_groups = align_multiple_datasets(*aligned_metadata_iterable)
    return all_metadata_groups


@fixture
def epic_kitchens_grouped_metadata(
    epic_kitchens_metadata_parser: EpicKitchensMetadataParser, progress: Progress
) -> Iterator[list[DatasetMetadata]]:
    return (
        [epic_kitchens_metadata_parser.convert_to_dataset_metadata(metadata)]
        for metadata in epic_kitchens_metadata_parser.get_metadata(progress)
    )


@fixture
def alfred_grouped_metadata(
    alfred_metadata_parser: AlfredMetadataParser, progress: Progress
) -> Iterator[list[DatasetMetadata]]:
    alfred_metadata = alfred_metadata_parser.get_metadata(progress)
    dataset_metadata_iterator = itertools.chain.from_iterable(
        alfred_metadata_parser.convert_to_dataset_metadata(metadata)
        for metadata in alfred_metadata
    )
    dataset_metadata = ([metadata] for metadata in dataset_metadata_iterator)

    return dataset_metadata


@fixture
def all_grouped_metadata(
    vg_coco_gqa_grouped_metadata: Iterator[list[DatasetMetadata]],
    epic_kitchens_grouped_metadata: Iterator[list[DatasetMetadata]],
    alfred_grouped_metadata: Iterator[list[DatasetMetadata]],
) -> Iterator[list[DatasetMetadata]]:
    return itertools.chain(
        vg_coco_gqa_grouped_metadata, epic_kitchens_grouped_metadata, alfred_grouped_metadata
    )


INSTANCES_DB_PATH = Path(Settings().paths.storage.joinpath("fixtures", "db", "instances.db"))


@fixture
@parametrize("instances_db_path", [pytest.param(INSTANCES_DB_PATH, id="subset")])
def instances_db(instances_db_path: Path) -> DatasetDb:
    with DatasetDb(instances_db_path) as db:
        yield db
