from multiprocessing.pool import Pool
from typing import Any, Callable, Optional

from rich.progress import Progress
from rich_click import typer

from emma_datasets.common import Settings, get_progress
from emma_datasets.datamodels import AnnotationType, DatasetName
from emma_datasets.parsers.annotation_extractors import (
    AlfredCaptionExtractor,
    AlfredTaskDescriptionExtractor,
    AlfredTrajectoryExtractor,
    AnnotationExtractor,
    CocoCaptionExtractor,
    ConceptualCaptionsExtractor,
    EpicKitchensCaptionExtractor,
    GqaQaPairExtractor,
    GqaSceneGraphExtractor,
    VgRegionsExtractor,
    VQAv2QaPairExtractor,
)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    short_help="Extract annotations from datasets.",
)


settings = Settings()
settings.paths.create_dirs()


def extract_coco_captions(progress: Progress) -> CocoCaptionExtractor:
    """Extract captions from COCO."""
    return CocoCaptionExtractor(
        [
            settings.paths.coco.joinpath("captions_train2017.json"),
            settings.paths.coco.joinpath("captions_val2017.json"),
        ],
        settings.paths.captions,
        progress,
    )


def extract_gqa_qa_pairs(progress: Progress) -> GqaQaPairExtractor:
    """Extract QA Pairs from GQA."""
    return GqaQaPairExtractor(
        [
            settings.paths.gqa_questions.joinpath("val_balanced_questions.json"),
            settings.paths.gqa_questions.joinpath("train_balanced_questions.json"),
        ],
        settings.paths.qa_pairs,
        progress,
    )


def extract_vqa_v2_qa_pairs(progress: Progress) -> VQAv2QaPairExtractor:
    """Extract QA Pairs from GQA."""
    return VQAv2QaPairExtractor(
        [
            (
                settings.paths.vqa_v2.joinpath("v2_OpenEnded_mscoco_val2014_questions.json"),
                settings.paths.vqa_v2.joinpath("v2_mscoco_val2014_annotations.json"),
            ),
            (
                settings.paths.vqa_v2.joinpath("VG_questions.json"),
                settings.paths.vqa_v2.joinpath("VG_annotations.json"),
            ),
            (
                settings.paths.vqa_v2.joinpath("v2_OpenEnded_mscoco_train2014_questions.json"),
                settings.paths.vqa_v2.joinpath("v2_mscoco_train2014_annotations.json"),
            ),
        ],
        settings.paths.qa_pairs,
        progress,
    )


def extract_gqa_scene_graphs(progress: Progress) -> GqaSceneGraphExtractor:
    """Extract scene graphs from GQA."""
    return GqaSceneGraphExtractor(
        [
            settings.paths.gqa_scene_graphs.joinpath("train_sceneGraphs.json"),
            settings.paths.gqa_scene_graphs.joinpath("val_sceneGraphs.json"),
        ],
        settings.paths.scene_graphs,
        progress,
    )


def extract_vg_regions(progress: Progress) -> VgRegionsExtractor:
    """Extract regions from VisualGenome."""
    return VgRegionsExtractor(
        settings.paths.visual_genome.joinpath("region_descriptions.json"),
        settings.paths.regions,
        progress,
    )


def extract_epic_kitchen_captions(progress: Progress) -> EpicKitchensCaptionExtractor:
    """Extract captions from EPIC-KITCHENS."""
    return EpicKitchensCaptionExtractor(
        [
            settings.paths.epic_kitchens.joinpath("EPIC_100_train.csv"),
            settings.paths.epic_kitchens.joinpath("EPIC_100_validation.csv"),
        ],
        settings.paths.captions,
        progress,
    )


def extract_alfred_captions(progress: Progress) -> AlfredCaptionExtractor:
    """Extract captions from ALFRED."""
    return AlfredCaptionExtractor(
        [
            settings.paths.alfred_data.joinpath("train/"),
            settings.paths.alfred_data.joinpath("valid_seen/"),
        ],
        settings.paths.captions,
        progress,
    )


def extract_alfred_subgoal_trajectories(progress: Progress) -> AlfredTrajectoryExtractor:
    """Extract subgoal trajectories from ALFRED."""
    return AlfredTrajectoryExtractor(
        [
            settings.paths.alfred_data.joinpath("train/"),
            settings.paths.alfred_data.joinpath("valid_seen/"),
        ],
        settings.paths.trajectories,
        progress,
    )


def extract_alfred_task_descriptions(progress: Progress) -> AlfredTaskDescriptionExtractor:
    """Extract task descriptions from ALFRED."""
    return AlfredTaskDescriptionExtractor(
        [
            settings.paths.alfred_data.joinpath("train"),
            settings.paths.alfred_data.joinpath("valid_seen"),
        ],
        settings.paths.task_descriptions,
        progress,
    )


def extract_conceptual_captions(progress: Progress) -> ConceptualCaptionsExtractor:
    """Extract captions from Conceptual Captions."""
    return ConceptualCaptionsExtractor(
        [
            settings.paths.conceptual_captions.joinpath("train/"),
            settings.paths.conceptual_captions.joinpath("valid/"),
        ],
        settings.paths.captions.joinpath("conceptual_captions"),
        progress,
    )


all_extractor_callables: list[Callable[[Progress], AnnotationExtractor[Any]]] = [
    extract_coco_captions,
    extract_gqa_qa_pairs,
    extract_gqa_scene_graphs,
    extract_vg_regions,
    extract_epic_kitchen_captions,
    extract_alfred_captions,
    extract_alfred_subgoal_trajectories,
    extract_alfred_task_descriptions,
    extract_conceptual_captions,
    extract_vqa_v2_qa_pairs,
]


@app.command("annotations")
def extract_annotation_by_type(
    annotations: Optional[list[AnnotationType]] = typer.Option(  # noqa: WPS404
        None,
        case_sensitive=False,
        help="Optionally, specify which annotation types to extract from the various datasets.",
    ),
    datasets: Optional[list[DatasetName]] = typer.Option(  # noqa: WPS404
        None,
        case_sensitive=False,
        help="Optionally, specify which datasets to extract annotations from.",
    ),
    num_workers: Optional[int] = typer.Option(  # noqa: WPS404
        None, show_default=False, help="Use maximum available workers by default."
    ),
) -> None:
    """Extract annotations from various datasets.

    By default, all annotations across all datasets are extracted, using the maximum available
    workers.
    """
    progress = get_progress()

    extractors = [extractor(progress) for extractor in all_extractor_callables]

    # Remove any extractors that do not extract the specified annotation types
    if annotations:
        extractors = [
            extractor for extractor in extractors if extractor.annotation_type in annotations
        ]

    # Remove any extractors that do not support the specified datasets
    if datasets:
        extractors = [extractor for extractor in extractors if extractor.dataset_name in datasets]

    # Error if there are no extractors left
    if not extractors:
        progress.console.log(
            "[b red]ERROR:[/] No extractors are available for the given datasets and annotation types."
        )
        raise typer.Abort()

    with progress:
        with Pool(num_workers) as pool:
            for extractor in extractors:
                extractor.run(progress, pool)


if __name__ == "__main__":
    app()
