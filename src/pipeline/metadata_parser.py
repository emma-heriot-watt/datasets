import itertools
from multiprocessing.pool import Pool
from typing import Iterator, Optional

from rich.progress import Progress

from src.common import Settings
from src.datamodels import DatasetMetadata, DatasetName
from src.datamodels.datasets import CocoImageMetadata, GqaImageMetadata, VgImageMetadata
from src.parsers.align_multiple_datasets import AlignMultipleDatasets
from src.parsers.dataset_aligner import DatasetAligner
from src.parsers.dataset_metadata import CocoMetadataParser, GqaMetadataParser, VgMetadataParser


settings = Settings()


class MetadataParser:
    """Provide a simple interface for parsing metadata for all the datasets."""

    def __init__(self, progress: Progress) -> None:
        self.progress = progress

        self._vg = VgMetadataParser(
            settings.paths.visual_genome.joinpath("image_data.json"),
            images_dir=settings.paths.visual_genome.joinpath("images/"),
            regions_dir=settings.paths.regions,
            progress=self.progress,
        )

        self._gqa = GqaMetadataParser(
            scene_graphs_train_path=settings.paths.gqa.joinpath("train_sceneGraphs.json"),
            scene_graphs_val_path=settings.paths.gqa.joinpath("val_sceneGraphs.json"),
            images_dir=settings.paths.gqa.joinpath("images/"),
            scene_graphs_dir=settings.paths.scene_graphs,
            qa_pairs_dir=settings.paths.qa_pairs,
            progress=progress,
        )

        self._coco = CocoMetadataParser(
            caption_train_path=settings.paths.coco.joinpath("captions_train2017.json"),
            caption_val_path=settings.paths.coco.joinpath("captions_val2017.json"),
            images_dir=settings.paths.coco.joinpath("images/"),
            captions_dir=settings.paths.captions,
            progress=progress,
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
        )

    def coco_vg_gqa(self, pool: Optional[Pool] = None) -> Iterator[list[DatasetMetadata]]:
        """Get groups of aligned dataset metadata from COCO, VG, and GQA."""
        aligned_vg_coco_metadata = self._vg_coco_aligner.get_aligned_metadata(pool)
        aligned_gqa_vg_metadata = self._gqa_vg_aligner.get_aligned_metadata(pool)

        dataset_metadata = self._align_coco_gqa_with_vg(
            aligned_vg_coco_metadata, aligned_gqa_vg_metadata
        )
        return dataset_metadata
