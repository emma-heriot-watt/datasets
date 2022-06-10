import itertools
from multiprocessing.pool import Pool
from typing import Iterator, Optional

from rich.progress import Progress

from emma_datasets.common import Settings
from emma_datasets.datamodels import DatasetMetadata, DatasetName, DatasetSplit
from emma_datasets.datamodels.datasets import CocoImageMetadata, GqaImageMetadata, VgImageMetadata
from emma_datasets.parsers.align_multiple_datasets import AlignMultipleDatasets
from emma_datasets.parsers.dataset_aligner import DatasetAligner
from emma_datasets.parsers.dataset_metadata import (
    AlfredMetadataParser,
    CocoMetadataParser,
    EpicKitchensMetadataParser,
    GqaMetadataParser,
    VgMetadataParser,
)


settings = Settings()


class MetadataParser:
    """Provide a simple interface for parsing metadata for all the datasets."""

    def __init__(self, progress: Progress) -> None:
        self.progress = progress

        self._vg = VgMetadataParser(
            settings.paths.visual_genome.joinpath("image_data.json"),
            images_dir=settings.paths.visual_genome_images,
            regions_dir=settings.paths.regions,
            features_dir=settings.paths.visual_genome_features,
            progress=self.progress,
        )

        self._gqa = GqaMetadataParser(
            scene_graphs_train_path=settings.paths.gqa_scene_graphs.joinpath(
                "train_sceneGraphs.json"
            ),
            scene_graphs_val_path=settings.paths.gqa_scene_graphs.joinpath("val_sceneGraphs.json"),
            images_dir=settings.paths.gqa_images,
            scene_graphs_dir=settings.paths.scene_graphs,
            qa_pairs_dir=settings.paths.qa_pairs,
            features_dir=settings.paths.gqa_features,
            progress=self.progress,
        )

        self._coco = CocoMetadataParser(
            caption_train_path=settings.paths.coco.joinpath("captions_train2017.json"),
            caption_val_path=settings.paths.coco.joinpath("captions_val2017.json"),
            images_dir=settings.paths.coco_images,
            captions_dir=settings.paths.captions,
            features_dir=settings.paths.coco_features,
            progress=self.progress,
        )

        self._epic_kitchens = EpicKitchensMetadataParser(
            data_paths=[
                (settings.paths.epic_kitchens.joinpath("EPIC_100_train.csv"), DatasetSplit.train),
                (
                    settings.paths.epic_kitchens.joinpath("EPIC_100_validation.csv"),
                    DatasetSplit.valid,
                ),
            ],
            frames_dir=settings.paths.epic_kitchens_frames,
            captions_dir=settings.paths.captions,
            features_dir=settings.paths.epic_kitchens_features,
            video_info_file=settings.paths.epic_kitchens.joinpath("EPIC_100_video_info.csv"),
            progress=self.progress,
        )

        self._alfred = AlfredMetadataParser(
            data_paths=[
                (settings.paths.alfred_data.joinpath("train/"), DatasetSplit.train),
                (settings.paths.alfred_data.joinpath("valid_seen/"), DatasetSplit.valid),
            ],
            alfred_dir=settings.paths.alfred_data,
            captions_dir=settings.paths.captions,
            trajectories_dir=settings.paths.trajectories,
            features_dir=settings.paths.alfred_features,
            task_descriptions_dir=settings.paths.task_descriptions,
            progress=self.progress,
        )

        self._vg_coco_aligner = DatasetAligner[VgImageMetadata, CocoImageMetadata](
            self._vg,
            self._coco,
            source_mapping_attr_for_target="coco_id",
            target_mapping_attr_for_source="id",
            progress=self.progress,
        )

        self._gqa_vg_aligner = DatasetAligner[GqaImageMetadata, VgImageMetadata](
            self._gqa,
            self._vg,
            source_mapping_attr_for_target="id",
            target_mapping_attr_for_source="image_id",
            progress=self.progress,
        )

        self._align_coco_gqa_with_vg = AlignMultipleDatasets(
            DatasetName.visual_genome, self.progress, "Merging VG, COCO and GQA where possible"
        )

    def get_all_metadata_groups(
        self, pool: Optional[Pool] = None
    ) -> Iterator[list[DatasetMetadata]]:
        """Get all dataset metadata from the input datasets."""
        return itertools.chain(
            self.coco_vg_gqa(pool),
            self.epic_kitchens(pool),
            self.alfred(pool),
        )

    def coco_vg_gqa(self, pool: Optional[Pool] = None) -> Iterator[list[DatasetMetadata]]:
        """Get groups of aligned dataset metadata from COCO, VG, and GQA."""
        aligned_vg_coco_metadata = self._vg_coco_aligner.get_aligned_metadata(pool)
        aligned_gqa_vg_metadata = self._gqa_vg_aligner.get_aligned_metadata(pool)

        dataset_metadata = self._align_coco_gqa_with_vg(
            aligned_vg_coco_metadata, aligned_gqa_vg_metadata
        )
        return dataset_metadata

    def epic_kitchens(self, pool: Optional[Pool] = None) -> Iterator[list[DatasetMetadata]]:
        """Get dataset metadata from the EPIC-KITCHENS dataset."""
        narration_metadata = self._epic_kitchens.get_metadata(self.progress, pool)
        dataset_metadata = (
            [self._epic_kitchens.convert_to_dataset_metadata(metadata)]
            for metadata in narration_metadata
        )

        return dataset_metadata

    def alfred(self, pool: Optional[Pool] = None) -> Iterator[list[DatasetMetadata]]:
        """Get dataset metadata from the ALFRED dataset."""
        alfred_metadata = self._alfred.get_metadata(self.progress, pool)
        dataset_metadata_iterator = itertools.chain.from_iterable(
            self._alfred.convert_to_dataset_metadata(metadata) for metadata in alfred_metadata
        )
        dataset_metadata = ([metadata] for metadata in dataset_metadata_iterator)

        return dataset_metadata
