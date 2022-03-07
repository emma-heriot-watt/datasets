from multiprocessing.pool import Pool
from typing import Iterable

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
def test_instance_creator_works(
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
