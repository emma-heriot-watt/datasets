from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional

from rich.progress import Progress

from src.common import get_progress
from src.parsers.instance_splitters import (
    CocoCaptionSplitter,
    GqaQaPairSplitter,
    GqaSceneGraphSplitter,
    VgRegionsSplitter,
)


BASE_DIR = Path("storage/data")
CAPTIONS_DIR = BASE_DIR.joinpath("captions").as_posix()
QA_PAIRS_DIR = BASE_DIR.joinpath("qa_pairs").as_posix()
SCENE_GRAPH_DIR = BASE_DIR.joinpath("scene_graphs").as_posix()
REGIONS_DIR = BASE_DIR.joinpath("regions").as_posix()


def split_dataset_instances(num_workers: int = 4, progress: Optional[Progress] = None) -> None:
    """Split dataset instances into multiple files for faster merging later."""
    progress = progress if progress else get_progress()

    with progress:
        coco_captions = CocoCaptionSplitter(
            [
                "storage/data/coco/captions_train2017.json",
                "storage/data/coco/captions_val2017.json",
            ],
            CAPTIONS_DIR,
            progress,
        )

        gqa_qa_pairs = GqaQaPairSplitter(
            [
                "storage/data/gqa/questions/val_balanced_questions.json",
                "storage/data/gqa/questions/train_balanced_questions.json",
            ],
            QA_PAIRS_DIR,
            progress,
        )

        gqa_scene_graph = GqaSceneGraphSplitter(
            [
                "storage/data/gqa/train_sceneGraphs.json",
                "storage/data/gqa/val_sceneGraphs.json",
            ],
            SCENE_GRAPH_DIR,
            progress,
        )

        vg_regions = VgRegionsSplitter(
            "storage/data/visual_genome/region_descriptions.json",
            REGIONS_DIR,
            progress,
        )

        with Pool(num_workers) as pool:
            coco_captions.run(progress, pool)
            gqa_qa_pairs.run(progress, pool)
            gqa_scene_graph.run(progress, pool)
            vg_regions.run(progress, pool)


if __name__ == "__main__":
    split_dataset_instances(8)
