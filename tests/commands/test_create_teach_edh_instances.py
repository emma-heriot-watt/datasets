from pathlib import Path

from pytest_cases import fixture_ref, parametrize
from rich.progress import Progress

from emma_datasets.commands.create_teach_edh_instances import (
    TeachEdhInstanceDbCreator,
    create_teach_edh_instances,
)
from emma_datasets.datamodels import DatasetSplit, TeachEdhInstance
from emma_datasets.db import DatasetDb


@parametrize(
    "edh_instances_data_paths, dataset_split",
    [
        [fixture_ref("teach_edh_train_data_paths"), DatasetSplit.train],
        [fixture_ref("teach_edh_valid_seen_data_paths"), DatasetSplit.valid_seen],
        [fixture_ref("teach_edh_valid_unseen_data_paths"), DatasetSplit.valid_unseen],
    ],
    ids=["train", "valid_seen", "valid_unseen"],
)
def test_teach_db_creator_writes_instances_that_can_be_parsed(
    edh_instances_data_paths: list[Path],
    dataset_split: DatasetSplit,
    tmp_path: Path,
    progress: Progress,
) -> None:
    creator = TeachEdhInstanceDbCreator(
        edh_instance_file_paths=edh_instances_data_paths,
        dataset_split=dataset_split,
        progress=progress,
        output_dir=tmp_path,
    )
    creator.run()

    with DatasetDb(creator.db_path, readonly=True) as read_db:
        assert len(read_db)

        for _, _, instance_str in read_db:
            new_instance = TeachEdhInstance.parse_raw(instance_str)
            assert isinstance(new_instance, TeachEdhInstance)


def test_can_create_all_edh_instance_dbs_in_one_go(
    teach_edh_instance_path: Path, tmp_path: Path, progress: Progress
) -> None:
    create_teach_edh_instances(
        teach_edh_instances_splits_path=teach_edh_instance_path,
        output_dir=tmp_path,
        progress=progress,
    )

    # Ensure there are 3 db files that have been created
    assert len(list(tmp_path.iterdir())) == 3

    for db_path in tmp_path.iterdir():
        assert db_path.is_file()
        assert db_path.suffix.endswith("db")

        read_db = DatasetDb(db_path, readonly=True)

        assert len(read_db)

        for _, _, instance_str in read_db:
            new_instance = TeachEdhInstance.parse_raw(instance_str)
            assert isinstance(new_instance, TeachEdhInstance)
