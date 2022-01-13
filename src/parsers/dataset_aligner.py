from typing import Generic, Iterable, NamedTuple, TypeVar

from pydantic import BaseModel
from rich.progress import Progress
from rich.table import Table

from src.common import get_logger
from src.datamodels import DatasetMetadata, DatasetName
from src.parsers.dataset_metadata import DatasetMetadataParser


log = get_logger(__name__)


T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=BaseModel)


class DatasetAlignerReturn(NamedTuple):
    """Tuple of aligned and non-aligned instances."""

    aligned: list[dict[DatasetName, DatasetMetadata]]
    non_aligned: list[dict[DatasetName, DatasetMetadata]]


class DatasetAligner(Generic[S, T]):
    """Align two datasets together using their metadata."""

    def __init__(
        self,
        source_metadata_parser: DatasetMetadataParser[S],
        target_metadata_parser: DatasetMetadataParser[T],
        source_mapping_attr_for_target: str,
        target_mapping_attr_for_source: str,
        progress: Progress,
    ) -> None:
        self.source_metadata_parser = source_metadata_parser
        self.target_metadata_parser = target_metadata_parser

        self.source_mapping_attr_for_target = source_mapping_attr_for_target
        self.target_mapping_attr_for_source = target_mapping_attr_for_source

        self._source_dataset_name = self.source_metadata_parser.dataset_name.value
        self._target_dataset_name = self.target_metadata_parser.dataset_name.value

        self._source_dataset_size = 0
        self._target_dataset_size = 0

        self.progress = progress
        self.task_id = progress.add_task(
            description=f"Aligning {self._source_dataset_name} with {self._target_dataset_name}",
            start=False,
            visible=False,
        )

    def get_aligned_metadata(self) -> DatasetAlignerReturn:
        """Align and return the metadata for the two datasets."""
        source_metadata = self.source_metadata_parser.get_metadata()
        target_metadata = self.target_metadata_parser.get_metadata()

        aligned, non_aligned = self.align_all_instances(source_metadata, target_metadata)

        self._print_statistics(aligned)

        return DatasetAlignerReturn(aligned=aligned, non_aligned=non_aligned)

    def align_all_instances(
        self, source_instances: Iterable[S], target_instances: Iterable[T]
    ) -> DatasetAlignerReturn:
        """Align all instances in the source and target datasets.

        Args:
            source_instances (Iterable[S]):
                Instance metadata from source dataset
            target_instances (Iterable[T]):
                Instance metadata from target dataset

        Returns:
            DatasetAlignerReturn: Named tuple with lists of aligned and non-aligned instances.
        """
        aligned = []
        non_aligned = []

        target_mapping_for_source = self.get_target_mapping_for_source(source_instances)
        alignable_target_metadata = self.get_alignable_target_metadata(target_instances)

        self.progress.reset(self.task_id, visible=True, start=True)
        progress_bar = self.progress.track(
            alignable_target_metadata, total=len(alignable_target_metadata), task_id=self.task_id
        )

        for target_instance in progress_bar:
            aligned_target, non_aligned_target = self.align_target_instance_to_source(
                target_instance, target_mapping_for_source
            )

            if any(aligned_target):
                aligned.append(aligned_target)
            elif any(non_aligned_target):
                non_aligned.append(non_aligned_target)

        if non_aligned:
            log.warning(
                "{incorrect_length:,} instances from {target_name} do not have a valid ID for {source_name}. These additional instances are [bold red]not[/] ignored, but maybe you want to have a look at them?".format(
                    incorrect_length=len(non_aligned),
                    target_name=self._target_dataset_name,
                    source_name=self._source_dataset_name,
                )
            )

        return DatasetAlignerReturn(aligned=aligned, non_aligned=non_aligned)

    def align_target_instance_to_source(
        self, target_instance: T, target_mapping_for_source: dict[str, S]
    ) -> tuple[dict[DatasetName, DatasetMetadata], ...]:
        """Align single target instance to the source dataset if possible."""
        aligned = {}
        non_aligned = {}

        target_key_for_source = getattr(target_instance, self.target_mapping_attr_for_source)
        target_metadata = self.target_metadata_parser.convert_to_dataset_metadata(target_instance)

        try:
            source_instance = target_mapping_for_source[target_key_for_source]
        except KeyError:
            non_aligned = {self.target_metadata_parser.dataset_name: target_metadata}
        else:
            source_metadata = self.source_metadata_parser.convert_to_dataset_metadata(
                source_instance
            )
            aligned = {
                self.target_metadata_parser.dataset_name: target_metadata,
                self.source_metadata_parser.dataset_name: source_metadata,
            }

        return aligned, non_aligned

    def get_alignable_target_metadata(self, target_metadata: Iterable[T]) -> list[T]:
        """Get all target metadata which can be aligned to the source.

        This also updates the `_target_dataset_size` attribute.
        """
        target_metadata = list(target_metadata)
        self._target_dataset_size = len(target_metadata)

        alignable_target_metadata = [
            metadata
            for metadata in target_metadata
            if getattr(metadata, self.target_mapping_attr_for_source, None) is not None
        ]

        return alignable_target_metadata

    def get_target_mapping_for_source(self, source_instances: Iterable[S]) -> dict[str, S]:
        """Map each source instance to a target instance using the target attribute.

        This also updates the `_source_dataset_size` attribute.
        """
        source_instances = list(source_instances)
        self._source_dataset_size = len(source_instances)

        mapped_metadata = {
            getattr(metadata, self.source_mapping_attr_for_target): metadata
            for metadata in source_instances
            if getattr(metadata, self.source_mapping_attr_for_target, None) is not None
        }

        if self._source_dataset_size != len(mapped_metadata.items()):
            log.warning(
                "Not all scenes from {source_name} have an ID for {target_name}. {source_name} has a total of {source_metadata_length:,} instances, but only {mapped_metadata_length:,} have an ID for {target_name}.".format(
                    source_name=self._source_dataset_name,
                    target_name=self._target_dataset_name,
                    source_metadata_length=self._source_dataset_size,
                    mapped_metadata_length=len(mapped_metadata.items()),
                )
            )

        return mapped_metadata

    def _print_statistics(self, aligned: Iterable[dict[DatasetName, DatasetMetadata]]) -> None:
        table = Table(
            title=f"Alignment Stats for {self._source_dataset_name} and {self._target_dataset_name}",
        )
        table.add_column("Description")
        table.add_column("Count", justify="right", style="green")

        source_metadata_count = self._source_dataset_size
        target_metadata_count = self._target_dataset_size
        total_instance_count = source_metadata_count + target_metadata_count

        aligned_count = len(list(aligned))
        non_aligned_from_source = source_metadata_count - aligned_count
        non_aligned_from_target = target_metadata_count - aligned_count

        table.add_row(
            f"Instances from [u]{self._source_dataset_name}[/]",
            f"{source_metadata_count:,}",
        )
        table.add_row(
            f"Instances from [u]{self._target_dataset_name}[/]",
            f"{target_metadata_count:,}",
        )
        table.add_row("Total instances", f"{total_instance_count:,}", end_section=True)

        table.add_row("Aligned", f"{aligned_count:,}")
        table.add_row(
            f"Instances from [u]{self._source_dataset_name}[/] that can't be aligned",
            f"{(non_aligned_from_source):,}",
        )
        table.add_row(
            f"Instances from [u]{self._target_dataset_name}[/] that can't be aligned",
            f"{(non_aligned_from_target):,}",
            end_section=True,
        )

        total_instances_after_aligned = (
            (aligned_count * 2) + non_aligned_from_source + non_aligned_from_target
        )

        if total_instance_count > total_instances_after_aligned:
            table.add_row(
                "Instances not accounted for after alignment", f"{total_instances_after_aligned:,}"
            )

        self.progress.console.print(table, justify="center")
