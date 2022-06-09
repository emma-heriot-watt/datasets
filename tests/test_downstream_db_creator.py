from pathlib import Path

from emma_datasets.datamodels import DatasetName, DatasetSplit
from emma_datasets.datamodels.datasets import TeachEdhInstance
from emma_datasets.db import DatasetDb
from emma_datasets.pipeline import DownstreamDbCreator


def test_downstream_db_creator_works_with_teach_edh(
    teach_edh_train_data_paths: list[Path],
    teach_edh_valid_seen_data_paths: list[Path],
    teach_edh_valid_unseen_data_paths: list[Path],
    tmp_path: Path,
) -> None:
    paths_per_split = {
        DatasetSplit.train: teach_edh_train_data_paths,
        DatasetSplit.valid_seen: teach_edh_valid_seen_data_paths,
        DatasetSplit.valid_unseen: teach_edh_valid_unseen_data_paths,
    }

    creator = DownstreamDbCreator.from_one_instance_per_json(
        dataset_name=DatasetName.teach,
        paths_per_split=paths_per_split,
        instance_model_type=TeachEdhInstance,
        output_dir=tmp_path,
    )
    creator.run(num_workers=1)

    for dataset_split in paths_per_split.keys():
        with DatasetDb(creator._get_db_path(dataset_split), readonly=True) as read_db:
            assert len(read_db)

            for _, _, instance_str in read_db:
                new_instance = TeachEdhInstance.parse_raw(instance_str)
                assert isinstance(new_instance, TeachEdhInstance)
