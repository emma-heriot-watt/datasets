from pathlib import Path
from typing import Any

from overrides import overrides
from pydantic import parse_obj_as

from emma_datasets.datamodels import AnnotationType, Caption, DatasetName
from emma_datasets.io import get_all_file_paths, read_parquet
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class ConceptualCaptionsExtractor(AnnotationExtractor[Caption]):
    """Split Conceptual Captions into multiple files.

    Conceptual captions is downloaded using img2dataset https://github.com/emma-simbot/img2dataset.
    For each train/val split the dataset is split into multiple shards. Each shard has a separate
    .parquet file that contains all the annotations for each example in the shard.
    """

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.caption

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.conceptual_captions

    @property
    def file_ext(self) -> str:
        """The file extension of the raw data files."""
        return "parquet"

    def read(self, file_path: Path) -> Any:
        """Read the json file.

        Due to sharding for train and validation we also need to obtain the shard id and the split.
        For example, for file_path 'storage/datasets/cc3m/train/00159/001590146.jpg' the shard_id
        is 00159 and the split is train. These are then used to store the data in the output_dir
        keeping the sharding to avoid overflowing the file system.
        """
        data = read_parquet(file_path)
        shard_id = file_path.stem
        split = file_path.parents[0].name

        data = data.assign(split=split)
        data = data.assign(shard_id=shard_id)
        return data

    def convert(self, raw_instance: Any) -> list[Caption]:
        """Convert objects to the common Caption."""
        caption = raw_instance["caption"]
        caption_instance = parse_obj_as(list[Caption], [{"text": caption}])

        return caption_instance

    def process_single_instance(self, raw_instances: Any) -> None:
        """Process raw instance and write to file."""
        shard_out_dir = self.output_dir.joinpath(raw_instances.split[0], raw_instances.shard_id[0])
        shard_out_dir.mkdir(parents=True, exist_ok=True)
        for _, raw_instance in raw_instances.iterrows():
            caption_instance = self.convert(raw_instance)

            shard_out_file_path = Path(
                raw_instances.split[0], raw_instances.shard_id[0], raw_instance.key
            )
            self._write(caption_instance, str(shard_out_file_path))

    @overrides(check_signature=False)
    def _read(self) -> list[dict[str, Any]]:
        """Read all files and return a single Iterator over all of them."""
        return [
            self.process_raw_file_return(self.read(file_path)) for file_path in self.file_paths  # type: ignore[arg-type]
        ]

    def _get_all_file_paths(self) -> None:
        """Get all the file paths for the dataset and store in state."""
        self.file_paths = [
            path for path in get_all_file_paths(self._paths) if path.suffix.endswith(self.file_ext)
        ]
