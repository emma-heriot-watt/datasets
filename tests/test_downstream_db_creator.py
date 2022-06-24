from pathlib import Path

from emma_datasets.datamodels import DatasetName, DatasetSplit
from emma_datasets.datamodels.datasets import TeachEdhInstance, VQAv2Instance
from emma_datasets.datamodels.datasets.vqa_v2 import load_vqa_v2_annotations
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


def test_downstream_db_creator_works_with_vqa_v2(
    vqa_v2_train_data_path: tuple[Path, Path],
    vqa_v2_valid_data_path: tuple[Path, Path],
    vqa_v2_test_data_path: tuple[Path, None],
    tmp_path: Path,
) -> None:
    vqa_v2_dir_paths = {
        DatasetSplit.train: vqa_v2_train_data_path,
        DatasetSplit.valid_seen: vqa_v2_valid_data_path,
        DatasetSplit.valid_unseen: vqa_v2_test_data_path,
    }

    paths_per_split = {}
    for split, split_paths in vqa_v2_dir_paths.items():
        paths_per_split[split] = load_vqa_v2_annotations(
            questions_path=split_paths[0], answers_path=split_paths[1]
        )

    creator = DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.vqa_v2,
        paths_per_split=paths_per_split,
        instance_model_type=VQAv2Instance,
        output_dir=tmp_path,
    )

    creator.run(num_workers=1)

    for dataset_split in paths_per_split.keys():
        with DatasetDb(creator._get_db_path(dataset_split), readonly=True) as read_db:
            assert len(read_db)

            for _, _, instance_str in read_db:
                new_instance = VQAv2Instance.parse_raw(instance_str)
                assert isinstance(new_instance, VQAv2Instance)
