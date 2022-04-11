from multiprocessing.pool import Pool
from typing import Iterable, Iterator, cast

import pytest
from pytest_cases import parametrize
from rich.progress import Progress

from emma_datasets.datamodels import DatasetMetadata, Instance
from emma_datasets.parsers.instance_creators import PretrainInstanceCreator


@parametrize("should_compress", [False, True], ids=["without_compression", "with_compression"])
@parametrize(
    "use_pool",
    [
        pytest.param([False], id="without_multiprocessing"),
        pytest.param([True], marks=pytest.mark.multiprocessing, id="multiprocessing"),
    ],
)
def test_instance_creator_returns_correct_type(
    all_grouped_metadata: Iterable[list[DatasetMetadata]],
    progress: Progress,
    should_compress: bool,
    use_pool: bool,
) -> None:
    pool = Pool(2) if use_pool else None
    return_type = bytes if should_compress else Instance

    instance_creator = PretrainInstanceCreator(progress, should_compress=should_compress)
    instance_iterator = instance_creator(all_grouped_metadata, progress, pool)

    for instance in instance_iterator:
        assert isinstance(instance, return_type)


def test_instance_creator_populates_fields_correctly(
    all_grouped_metadata: Iterable[list[DatasetMetadata]],
    progress: Progress,
) -> None:
    instance_creator = PretrainInstanceCreator(progress)
    instance_iterator = instance_creator(all_grouped_metadata, progress)

    # Cast for mypy
    instance_iterator = cast(Iterator[Instance], instance_iterator)

    for instance in instance_iterator:
        for dataset_metadata in instance.dataset.values():
            if dataset_metadata.scene_graph_path is not None:
                assert instance.scene_graph

            if dataset_metadata.regions_path is not None:
                assert instance.regions

            if dataset_metadata.caption_path is not None:
                assert instance.captions

            if dataset_metadata.qa_pairs_path is not None:
                assert instance.qa_pairs

            if dataset_metadata.action_trajectory_path is not None:
                assert instance.trajectory
