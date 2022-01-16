from multiprocessing.pool import Pool
from typing import Optional

from rich.progress import Progress

from src.common import Settings, get_progress
from src.parsers.instance_splitters import (
    CocoCaptionSplitter,
    GqaQaPairSplitter,
    GqaSceneGraphSplitter,
    VgRegionsSplitter,
)


settings = Settings()


def split_dataset_instances(num_workers: int = 4, progress: Optional[Progress] = None) -> None:
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
                settings.paths.gqa.joinpath("questions/val_balanced_questions.json"),
                settings.paths.gqa.joinpath("questions/train_balanced_questions.json"),
            ],
            settings.paths.qa_pairs,
            progress,
        )

        gqa_scene_graph = GqaSceneGraphSplitter(
            [
                settings.paths.gqa.joinpath("train_sceneGraphs.json"),
                settings.paths.gqa.joinpath("val_sceneGraphs.json"),
            ],
            settings.paths.scene_graphs,
            progress,
        )

        vg_regions = VgRegionsSplitter(
            settings.paths.visual_genome.joinpath("region_descriptions.json"),
            settings.paths.regions,
            progress,
        )

        with Pool(num_workers) as pool:
            coco_captions.run(progress, pool)
            gqa_qa_pairs.run(progress, pool)
            gqa_scene_graph.run(progress, pool)
            vg_regions.run(progress, pool)


if __name__ == "__main__":
    split_dataset_instances(8)
