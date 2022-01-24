from multiprocessing.pool import Pool
from typing import Optional

from rich.progress import Progress

from src.common import Settings, get_progress
from src.parsers.instance_splitters import (
    CocoCaptionSplitter,
    EpicKitchensCaptionSplitter,
    GqaQaPairSplitter,
    GqaSceneGraphSplitter,
    VgRegionsSplitter,
)


settings = Settings()


def split_dataset_instances(
    num_workers: Optional[int] = None, progress: Optional[Progress] = None
) -> None:
    """Split dataset instances into multiple files for faster merging later."""
    progress = progress if progress else get_progress()

    with progress:
        coco_captions = CocoCaptionSplitter(
            [
                settings.paths.coco.joinpath("captions_train2017.json"),
                settings.paths.coco.joinpath("captions_val2017.json"),
            ],
            settings.paths.captions,
            progress,
        )

        gqa_qa_pairs = GqaQaPairSplitter(
            [
                settings.paths.gqa_questions.joinpath("val_balanced_questions.json"),
                settings.paths.gqa_questions.joinpath("train_balanced_questions.json"),
            ],
            settings.paths.qa_pairs,
            progress,
        )

        gqa_scene_graph = GqaSceneGraphSplitter(
            [
                settings.paths.gqa_scene_graphs.joinpath("train_sceneGraphs.json"),
                settings.paths.gqa_scene_graphs.joinpath("val_sceneGraphs.json"),
            ],
            settings.paths.scene_graphs,
            progress,
        )

        vg_regions = VgRegionsSplitter(
            settings.paths.visual_genome.joinpath("region_descriptions.json"),
            settings.paths.regions,
            progress,
        )

        ek_captions = EpicKitchensCaptionSplitter(
            [
                settings.paths.epic_kitchens.joinpath("EPIC_100_train.csv"),
                settings.paths.epic_kitchens.joinpath("EPIC_100_validation.csv"),
            ],
            settings.paths.captions,
            progress,
        )

        with Pool(num_workers) as pool:
            coco_captions.run(progress, pool)
            gqa_qa_pairs.run(progress, pool)
            gqa_scene_graph.run(progress, pool)
            vg_regions.run(progress, pool)
            ek_captions.run(progress, pool)


if __name__ == "__main__":
    split_dataset_instances()
