from multiprocessing.pool import Pool
from typing import Optional

from rich.progress import Progress

from src.api.storage import DatasetDB
from src.common import Settings, get_progress
from src.datamodels import DatasetName
from src.datamodels.datasets import CocoImageMetadata, GqaImageMetadata, VgImageMetadata
from src.parsers.align_multiple_datasets import AlignMultipleDatasets
from src.parsers.dataset_aligner import DatasetAligner
from src.parsers.dataset_metadata import CocoMetadataParser, GqaMetadataParser, VgMetadataParser
from src.parsers.structure_instances import StructureInstances


settings = Settings()


instances_db_path = settings.directories.databases.joinpath("instances.db")


def create_pretraining_instances(
    num_workers: int = 3, progress: Optional[Progress] = None
) -> None:
    """Create all the pretraining instances."""
    progress = progress if progress else get_progress()

    with progress:
        gqa_metadata_parser = GqaMetadataParser(
            scene_graphs_train_path="storage/data/gqa/train_sceneGraphs.json",
            scene_graphs_val_path="storage/data/gqa/val_sceneGraphs.json",
            images_dir="storage/data/gqa/images",
            scene_graphs_dir=settings.directories.scene_graphs.as_posix(),
            qa_pairs_dir=settings.directories.qa_pairs.as_posix(),
            progress=progress,
        )
        vg_metadata_parser = VgMetadataParser(
            "storage/data/visual_genome/image_data.json",
            images_dir="storage/data/visual_genome/images",
            regions_dir=settings.directories.regions.as_posix(),
            progress=progress,
        )
        coco_metadata_parser = CocoMetadataParser(
            caption_train_path="storage/data/coco/captions_train2017.json",
            caption_val_path="storage/data/coco/captions_val2017.json",
            images_dir="storage/data/coco/images",
            captions_dir=settings.directories.captions.as_posix(),
            progress=progress,
        )

        gqa_vg_aligner = DatasetAligner[GqaImageMetadata, VgImageMetadata](
            gqa_metadata_parser,
            vg_metadata_parser,
            source_mapping_attr_for_target="id",
            target_mapping_attr_for_source="image_id",
            progress=progress,
        )

        vg_coco_aligner = DatasetAligner[VgImageMetadata, CocoImageMetadata](
            vg_metadata_parser,
            coco_metadata_parser,
            source_mapping_attr_for_target="coco_id",
            target_mapping_attr_for_source="id",
            progress=progress,
        )

        scene_creator = AlignMultipleDatasets(
            DatasetName.visual_genome, progress, "Merging VG, COCO and GQA where possible"
        )

        structure_instances = StructureInstances(progress)

        aligned_gqa_vg_metadata = gqa_vg_aligner.get_aligned_metadata()
        aligned_vg_coco_metadata = vg_coco_aligner.get_aligned_metadata()

        scenes = scene_creator.get_aligned_scenes(
            [aligned_vg_coco_metadata, aligned_gqa_vg_metadata]
        )

        with DatasetDB(instances_db_path, readonly=False) as db:
            progress.update(structure_instances.task_id, filepath=instances_db_path)

            with Pool(num_workers) as pool:
                instances_iterator = structure_instances.from_scenes(scenes, progress, pool)

                for i, instance in enumerate(instances_iterator):
                    db[(i, f"pretrain_{i}")] = instance.json()


if __name__ == "__main__":
    create_pretraining_instances()
