import itertools
from collections import ChainMap
from typing import Iterable, Iterator

from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, DatasetName
from emma_datasets.parsers.dataset_aligner import DatasetAlignerReturn


CommonDatasetMapping = dict[str, dict[DatasetName, DatasetMetadata]]


class AlignMultipleDatasets:
    """Align multiple aligned datasets, returning grouped of metadata per instance."""

    def __init__(
        self,
        common_dataset: DatasetName,
        progress: Progress,
        description: str = "Merging datasets",
    ) -> None:
        self.common_dataset = common_dataset

        self.progress = progress
        self.task_id = progress.add_task(description, start=False, visible=False, comment="")

    def __call__(
        self, *aligned_metadata_iterable: DatasetAlignerReturn
    ) -> Iterator[list[DatasetMetadata]]:
        """Align metadata across the multiple datasets to a signle common dataset.

        Args:
            aligned_metadata_iterable (Iterable[DatasetAlignerReturn]): Iterable of any aligned
                datasets which can be aligned to the common dataset.

        Returns:
            Iterator[list[DatasetMetadata]]: Generator which yield groups of metadata which can
                form the basis of a new instance.
        """
        all_common_dataset_mapping = [
            self._get_mapping_to_common_dataset(aligned_metadata)
            for aligned_metadata in aligned_metadata_iterable
        ]

        aligned_common_ids = self.get_common_aligned_ids(all_common_dataset_mapping)
        non_aligned_common_ids = self.get_non_aligned_common_ids(
            all_common_dataset_mapping, aligned_common_ids
        )

        self.progress.reset(
            self.task_id,
            start=True,
            total=self._calculate_total(all_common_dataset_mapping, aligned_metadata_iterable),
            visible=True,
        )

        common_instances = self.get_all_common_instances(
            all_common_dataset_mapping, aligned_common_ids
        )
        non_common_instances = self.get_all_non_common_instances(
            all_common_dataset_mapping, aligned_common_ids
        )
        non_aligned_instances = self.get_all_non_alignable_instances(
            aligned_metadata_iterable, non_aligned_common_ids
        )

        return itertools.chain(common_instances, non_common_instances, non_aligned_instances)

    def get_all_common_instances(
        self, all_common_dataset_mapping: list[CommonDatasetMapping], aligned_ids: set[str]
    ) -> Iterator[list[DatasetMetadata]]:
        """Get all scenes which align across all datasets."""
        for aligned_id in aligned_ids:
            instance = dict(
                ChainMap(*(mapping[aligned_id] for mapping in all_common_dataset_mapping))
            ).values()

            self.progress.advance(self.task_id)
            yield list(instance)

    def get_all_non_common_instances(
        self, all_common_dataset_mapping: list[CommonDatasetMapping], aligned_ids: set[str]
    ) -> Iterator[list[DatasetMetadata]]:
        """Get all instances which cannot be aligned with the common dataset."""
        for mapping in all_common_dataset_mapping:
            non_overlapping_ids = set(mapping.keys()) - aligned_ids

            for non_common_id in non_overlapping_ids:
                self.progress.advance(self.task_id)
                yield list(mapping[non_common_id].values())

    def get_all_non_alignable_instances(
        self, aligned_metadata_iterable: Iterable[DatasetAlignerReturn], non_common_ids: set[str]
    ) -> Iterator[list[DatasetMetadata]]:
        """Get all instances which cannot be aligned."""
        non_aligned_metadata = (metadata.non_aligned for metadata in aligned_metadata_iterable)

        existing_common_dataset_ids = set() | non_common_ids

        for non_aligned in itertools.chain.from_iterable(non_aligned_metadata):
            if self.common_dataset in non_aligned:
                metadata_id = non_aligned[self.common_dataset].id  # noqa: WPS529

                if metadata_id in existing_common_dataset_ids:
                    continue

                existing_common_dataset_ids.add(metadata_id)

            self.progress.advance(self.task_id)
            yield list(non_aligned.values())

    def get_common_aligned_ids(
        self, all_common_dataset_mapping: list[CommonDatasetMapping]
    ) -> set[str]:
        """Get IDs of instances from the common dataset which are aligned across all datasets.

        The IDs returned are for the `self.common_dataset`.
        """
        return set.intersection(*[set(mapping.keys()) for mapping in all_common_dataset_mapping])

    def get_non_aligned_common_ids(
        self, all_common_dataset_mapping: list[CommonDatasetMapping], aligned_common_ids: set[str]
    ) -> set[str]:
        """Get instance IDs which are aligned to the common dataset but not across all datasets.

        In other words, this returns a set of IDs from the common dataset which ARE aligned to one
        other dataset, but NOT ALL of the other datasets.
        """
        all_non_overlapping_ids = [
            set(mapping.keys()) - aligned_common_ids for mapping in all_common_dataset_mapping
        ]
        return set.union(*all_non_overlapping_ids)

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
        """Calculate total number of instances that will be returned."""
        aligned_common_ids = self.get_common_aligned_ids(all_common_dataset_mapping)

        common_ids_aligned_to_other_dataset = self.get_non_aligned_common_ids(
            all_common_dataset_mapping, aligned_common_ids
        )

        common_ids_not_aligned_to_any_dataset = {
            metadata_dict[self.common_dataset].id
            for metadata_list in aligned_metadata_iterable
            for metadata_dict in metadata_list.non_aligned
            if self.common_dataset in metadata_dict
            and metadata_dict[self.common_dataset].id not in common_ids_aligned_to_other_dataset
        }

        other_instances_not_aligned_to_common_dataset = [
            metadata_dict
            for metadata_list in aligned_metadata_iterable
            for metadata_dict in metadata_list.non_aligned
            if self.common_dataset not in metadata_dict
        ]

        return (
            len(aligned_common_ids)
            + len(common_ids_aligned_to_other_dataset)
            + len(common_ids_not_aligned_to_any_dataset)
            + len(other_instances_not_aligned_to_common_dataset)
        )
