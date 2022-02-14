import itertools
import os
from pathlib import Path
from typing import Iterator, Union, cast

import pytest
from pytest_cases import fixture, parametrize
from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName, DatasetSplit
from emma_datasets.datamodels.datasets import CocoImageMetadata, GqaImageMetadata, VgImageMetadata
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
    return Progress()


# ----------------------------------- Cases ---------------------------------- #


@fixture
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


# ----------------------------- Metadata Parsers ----------------------------- #


@fixture
def coco_metadata_parser(paths: dict[str, Path], progress: Progress) -> CocoMetadataParser:
    fake_caption_dir = Path("caption")
    fake_image_dir = Path("images")

    return CocoMetadataParser(
        caption_train_path=paths["coco_caption_train"],
        caption_val_path=paths["coco_caption_val"],
        images_dir=fake_image_dir,
        captions_dir=fake_caption_dir,
        features_dir=Path("features"),
        progress=progress,
    )


@fixture
def vg_metadata_parser(paths: dict[str, Path], progress: Progress) -> VgMetadataParser:
    fake_images_dir = Path("images")
    fake_regions_dir = Path("regions")

    return VgMetadataParser(
        image_data_json_path=paths["vg_image_data"],
        images_dir=fake_images_dir,
        regions_dir=fake_regions_dir,
        features_dir=Path("features"),
        progress=progress,
    )


@fixture
def gqa_metadata_parser(paths: dict[str, Path], progress: Progress) -> GqaMetadataParser:
    return GqaMetadataParser(
        scene_graphs_train_path=paths["gqa_scene_graph_train"],
        scene_graphs_val_path=paths["gqa_scene_graph_val"],
        images_dir=Path("images"),
        scene_graphs_dir=Path("scene_graphs"),
        qa_pairs_dir=Path("qa_pairs"),
        features_dir=Path("features"),
        progress=progress,
    )


@fixture
def epic_kitchens_metadata_parser(
    paths: dict[str, Path], progress: Progress
) -> EpicKitchensMetadataParser:
    data_paths: list[DataPathTuple] = [
        (paths["ek_train"], DatasetSplit.train),
        (paths["ek_val"], DatasetSplit.valid),
    ]
    return EpicKitchensMetadataParser(
        data_paths=data_paths,
        frames_dir=Path("frames"),
        captions_dir=Path("captions"),
        features_dir=Path("features"),
        video_info_file=paths["ek_video_info"],
        progress=progress,
    )


@fixture
def alfred_metadata_parser(
    paths: dict[str, list[Path]], progress: Progress
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
        captions_dir=Path("captions"),
        trajectories_dir=Path("trajectories"),
        features_dir=Path("features"),
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
    all_metadata_groups = itertools.chain(align_multiple_datasets(*aligned_metadata_iterable))
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
    return (
        list(alfred_metadata_parser.convert_to_dataset_metadata(metadata))
        for metadata in alfred_metadata_parser.get_metadata(progress)
    )


@fixture
def all_grouped_metadata(
    vg_coco_gqa_grouped_metadata: Iterator[list[DatasetMetadata]],
    epic_kitchens_grouped_metadata: Iterator[list[DatasetMetadata]],
    alfred_grouped_metadata: Iterator[list[DatasetMetadata]],
) -> Iterator[list[DatasetMetadata]]:
    return itertools.chain(
        epic_kitchens_grouped_metadata, vg_coco_gqa_grouped_metadata, alfred_grouped_metadata
    )
