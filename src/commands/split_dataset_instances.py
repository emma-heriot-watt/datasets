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
                "storage/data/coco/captions_train2017.json",
                "storage/data/coco/captions_val2017.json",
            ],
            settings.directories.captions.as_posix(),
            progress,
        )

        gqa_qa_pairs = GqaQaPairSplitter(
            [
                "storage/data/gqa/questions/val_balanced_questions.json",
                "storage/data/gqa/questions/train_balanced_questions.json",
            ],
            settings.directories.qa_pairs.as_posix(),
            progress,
        )

        gqa_scene_graph = GqaSceneGraphSplitter(
            [
                "storage/data/gqa/train_sceneGraphs.json",
                "storage/data/gqa/val_sceneGraphs.json",
            ],
            settings.directories.scene_graphs.as_posix(),
            progress,
        )

        vg_regions = VgRegionsSplitter(
            "storage/data/visual_genome/region_descriptions.json",
            settings.directories.regions.as_posix(),
            progress,
        )

        with Pool(num_workers) as pool:
            coco_captions.run(progress, pool)
            gqa_qa_pairs.run(progress, pool)
            gqa_scene_graph.run(progress, pool)
            vg_regions.run(progress, pool)


if __name__ == "__main__":
    split_dataset_instances(8)
