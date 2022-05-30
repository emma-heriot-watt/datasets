from multiprocessing.pool import Pool
from typing import Optional

from emma_datasets.common import Settings, get_progress, use_rich_for_logging
from emma_datasets.parsers.annotation_extractors import (
    AlfredCaptionExtractor,
    AlfredSubgoalTrajectoryExtractor,
    CocoCaptionExtractor,
    EpicKitchensCaptionExtractor,
    GqaQaPairExtractor,
    GqaSceneGraphExtractor,
    VgRegionsExtractor,
)


use_rich_for_logging()

settings = Settings()
settings.paths.create_dirs()


def extract_annotations(num_workers: Optional[int] = None) -> None:
    """Extract annotations from all the datasets into multiple files for faster processing."""
    progress = get_progress()

    with progress:
        coco_captions = CocoCaptionExtractor(
            [
                settings.paths.coco.joinpath("captions_train2017.json"),
                settings.paths.coco.joinpath("captions_val2017.json"),
            ],
            settings.paths.captions,
            progress,
        )

        gqa_qa_pairs = GqaQaPairExtractor(
            [
                settings.paths.gqa_questions.joinpath("val_balanced_questions.json"),
                settings.paths.gqa_questions.joinpath("train_balanced_questions.json"),
            ],
            settings.paths.qa_pairs,
            progress,
        )

        gqa_scene_graph = GqaSceneGraphExtractor(
            [
                settings.paths.gqa_scene_graphs.joinpath("train_sceneGraphs.json"),
                settings.paths.gqa_scene_graphs.joinpath("val_sceneGraphs.json"),
            ],
            settings.paths.scene_graphs,
            progress,
        )

        vg_regions = VgRegionsExtractor(
            settings.paths.visual_genome.joinpath("region_descriptions.json"),
            settings.paths.regions,
            progress,
        )

        ek_captions = EpicKitchensCaptionExtractor(
            [
                settings.paths.epic_kitchens.joinpath("EPIC_100_train.csv"),
                settings.paths.epic_kitchens.joinpath("EPIC_100_validation.csv"),
            ],
            settings.paths.captions,
            progress,
        )

        alfred_captions = AlfredCaptionExtractor(
            [
                settings.paths.alfred_data.joinpath("train/"),
                settings.paths.alfred_data.joinpath("valid_seen/"),
            ],
            settings.paths.captions,
            progress,
        )

        alfred_subgoal_trajectories = AlfredSubgoalTrajectoryExtractor(
            [
                settings.paths.alfred_data.joinpath("train/"),
                settings.paths.alfred_data.joinpath("valid_seen/"),
            ],
            settings.paths.trajectories,
            progress,
        )

        with Pool(num_workers) as pool:
            coco_captions.run(progress, pool)
            gqa_qa_pairs.run(progress, pool)
            gqa_scene_graph.run(progress, pool)
            vg_regions.run(progress, pool)
            ek_captions.run(progress, pool)
            alfred_captions.run(progress, pool)
            alfred_subgoal_trajectories.run(progress, pool)


if __name__ == "__main__":
    extract_annotations()
