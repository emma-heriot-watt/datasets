import itertools
from pathlib import Path

from pytest_cases import fixture, parametrize
from rich.progress import Progress

from emma_datasets.common import Settings
from emma_datasets.datamodels import DatasetSplit
from emma_datasets.parsers.dataset_metadata import (
    AlfredMetadataParser,
    CocoMetadataParser,
    DataPathTuple,
    EpicKitchensMetadataParser,
    GqaMetadataParser,
    VgMetadataParser,
)


settings = Settings()


@fixture
def coco_metadata_parser(
    coco_captions_path: Path, extracted_annotations_paths: dict[str, Path], progress: Progress
) -> CocoMetadataParser:
    return CocoMetadataParser(
        caption_train_path=coco_captions_path,
        caption_val_path=coco_captions_path,
        images_dir=settings.paths.coco_images,
        captions_dir=extracted_annotations_paths["coco_captions"],
        features_dir=settings.paths.coco_features,
        progress=progress,
    )


@fixture
def vg_metadata_parser(
    vg_image_data_path: Path, extracted_annotations_paths: dict[str, Path], progress: Progress
) -> VgMetadataParser:
    return VgMetadataParser(
        image_data_json_path=vg_image_data_path,
        images_dir=settings.paths.visual_genome_images,
        regions_dir=extracted_annotations_paths["regions"],
        features_dir=settings.paths.visual_genome_features,
        progress=progress,
    )


@fixture
def gqa_metadata_parser(
    gqa_scene_graphs_path: Path, extracted_annotations_paths: dict[str, Path], progress: Progress
) -> GqaMetadataParser:
    return GqaMetadataParser(
        scene_graphs_train_path=gqa_scene_graphs_path,
        scene_graphs_val_path=gqa_scene_graphs_path,
        images_dir=settings.paths.gqa_images,
        scene_graphs_dir=extracted_annotations_paths["scene_graphs"],
        qa_pairs_dir=extracted_annotations_paths["qa_pairs"],
        features_dir=settings.paths.gqa_features,
        progress=progress,
    )


@fixture
def epic_kitchens_metadata_parser(
    ek_data_path: Path,
    ek_video_info_path: Path,
    extracted_annotations_paths: dict[str, Path],
    progress: Progress,
) -> EpicKitchensMetadataParser:
    data_paths: list[DataPathTuple] = [
        (ek_data_path, DatasetSplit.train),
        (ek_data_path, DatasetSplit.valid),
    ]
    return EpicKitchensMetadataParser(
        data_paths=data_paths,
        frames_dir=settings.paths.epic_kitchens_frames,
        captions_dir=extracted_annotations_paths["ek_captions"],
        features_dir=settings.paths.epic_kitchens_features,
        video_info_file=ek_video_info_path,
        progress=progress,
    )


@fixture
def alfred_metadata_parser(
    fixtures_root: Path,
    alfred_train_data_path: list[Path],
    alfred_valid_seen_data_path: list[Path],
    extracted_annotations_paths: dict[str, Path],
    progress: Progress,
) -> AlfredMetadataParser:
    data_paths: list[DataPathTuple] = list(
        itertools.chain.from_iterable(
            [
                zip(alfred_train_data_path, itertools.cycle([DatasetSplit.train])),
                zip(alfred_valid_seen_data_path, itertools.cycle([DatasetSplit.valid])),
            ]
        )
    )

    return AlfredMetadataParser(
        data_paths=data_paths,
        alfred_dir=fixtures_root.joinpath("alfred/"),
        captions_dir=extracted_annotations_paths["alfred_captions"],
        trajectories_dir=extracted_annotations_paths["trajectories"],
        features_dir=settings.paths.alfred_features,
        progress=progress,
    )


@fixture
@parametrize(
    "metadata_parser",
    [
        coco_metadata_parser,
        vg_metadata_parser,
        gqa_metadata_parser,
        epic_kitchens_metadata_parser,
        alfred_metadata_parser,
    ],
)
def metadata_parser(metadata_parser):
    return metadata_parser


@fixture
def alfred_annotations(fixtures_root: Path) -> dict[str, Path]:
    alfred_dir = fixtures_root.joinpath("alfred/")
    annotations_dict = {}
    for annotation_file in alfred_dir.rglob("*.json"):
        annotations_dict[annotation_file.parts[-2]] = annotation_file
    return annotations_dict
