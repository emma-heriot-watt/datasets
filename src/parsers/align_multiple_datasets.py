import itertools
from collections import ChainMap
from typing import Iterable, Iterator

from rich.progress import Progress

from src.common import get_logger
from src.datamodels import DatasetMetadata, DatasetName, MediaType, Scene
from src.parsers.dataset_aligner import DatasetAlignerReturn


CommonDatasetMapping = dict[str, dict[DatasetName, DatasetMetadata]]

log = get_logger(__name__)


class AlignMultipleDatasets:
    """Align multiple aligned datasets, returning scenes."""

    def __init__(
        self,
        common_dataset: DatasetName,
        progress: Progress,
        description: str = "Merging datasets",
    ) -> None:
        self.common_dataset = common_dataset
        self.progress = progress

        self.task_id = progress.add_task(description, start=False, visible=False)

    def get_aligned_scenes(
        self, aligned_metadata_iterable: Iterable[DatasetAlignerReturn]
    ) -> Iterator[Scene]:
        """Get scenes from all given datasets."""
        all_common_dataset_mapping = [
            self._get_mapping_to_common_dataset(aligned_metadata)
            for aligned_metadata in aligned_metadata_iterable
        ]

        aligned_ids = self.get_aligned_ids(all_common_dataset_mapping)

        self.progress.reset(
            self.task_id,
            start=True,
            total=self._calculate_total(all_common_dataset_mapping, aligned_metadata_iterable),
            visible=True,
        )

        common_scenes = self.get_all_common_scenes(all_common_dataset_mapping, aligned_ids)
        non_common_scenes = self.get_all_non_common_scenes(all_common_dataset_mapping, aligned_ids)
        non_aligned_scenes = self.get_all_non_aligned_scenes(aligned_metadata_iterable)

        return itertools.chain(common_scenes, non_common_scenes, non_aligned_scenes)

    def get_all_common_scenes(
        self, all_common_dataset_mapping: list[CommonDatasetMapping], aligned_ids: set[str]
    ) -> Iterator[Scene]:
        """Get all scenes which align across all datasets."""
        for aligned_id in aligned_ids:
            scene = Scene(
                media_type=MediaType.image,
                dataset=dict(
                    ChainMap(*(mapping[aligned_id] for mapping in all_common_dataset_mapping))
                ),
            )

            self.progress.advance(self.task_id)
            yield scene

    def get_all_non_common_scenes(
        self, all_common_dataset_mapping: list[CommonDatasetMapping], aligned_ids: set[str]
    ) -> Iterator[Scene]:
        """Get all scenes which cannot be aligned with the common dataset."""
        for mapping in all_common_dataset_mapping:
            non_overlapping_ids = set(mapping.keys()) - aligned_ids

            for non_common_id in non_overlapping_ids:
                scene = Scene(media_type=MediaType.image, dataset=mapping[non_common_id])

                self.progress.advance(self.task_id)
                yield scene

    def get_all_non_aligned_scenes(
        self, aligned_metadata_iterable: Iterable[DatasetAlignerReturn]
    ) -> Iterator[Scene]:
        """Get all scenes which cannot be aligned."""
        non_aligned_metadata = (metadata.non_aligned for metadata in aligned_metadata_iterable)

        for non_aligned in itertools.chain.from_iterable(non_aligned_metadata):
            scene = Scene(media_type=MediaType.image, dataset=non_aligned)

            self.progress.advance(self.task_id)
            yield scene

    def get_aligned_ids(self, all_common_dataset_mapping: list[CommonDatasetMapping]) -> set[str]:
        """Get IDs of instances from the common dataset which are aligned across all datasets."""
        return set.intersection(*[set(mapping.keys()) for mapping in all_common_dataset_mapping])

    def _get_mapping_to_common_dataset(
        self, aligned_metadata: DatasetAlignerReturn
    ) -> CommonDatasetMapping:
        """Get the mapping that connects each instance to the common dataset."""
        return {
            metadata[self.common_dataset].id: metadata for metadata in aligned_metadata.aligned
        }

    def _calculate_total(
        self,
        all_common_dataset_mapping: list[CommonDatasetMapping],
        aligned_metadata_iterable: Iterable[DatasetAlignerReturn],
    ) -> int:
        aligned_ids = self.get_aligned_ids(all_common_dataset_mapping)
        non_overlapping_ids = [
            len(set(mapping.keys()) - aligned_ids) for mapping in all_common_dataset_mapping
        ]
        non_aligned_metadata = (
            len(metadata.non_aligned) for metadata in aligned_metadata_iterable
        )
        return len(aligned_ids) + sum(non_overlapping_ids) + sum(non_aligned_metadata)
