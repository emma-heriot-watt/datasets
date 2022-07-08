from itertools import groupby
from pathlib import Path
from typing import Optional, Union

import numpy as np
from rich_click import typer

from emma_datasets.common import Settings
from emma_datasets.datamodels import DatasetName, DatasetSplit
from emma_datasets.datamodels.datasets import (
    CocoInstance,
    TeachEdhInstance,
    VQAv2Instance,
    WinogroundInstance,
)
from emma_datasets.datamodels.datasets.nlvr import NlvrInstance
from emma_datasets.datamodels.datasets.refcoco import RefCocoInstance, load_refcoco_annotations
from emma_datasets.datamodels.datasets.vqa_v2 import (
    get_vqa_v2_annotation_paths,
    load_vqa_v2_annotations,
)
from emma_datasets.io import read_json
from emma_datasets.pipeline import DownstreamDbCreator


settings = Settings()


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    short_help="Create DBs for downstream datasets.",
    help="Create DBs for datasets that are being used for downstream evaluation of the model.",
)


@app.callback()
def callback() -> None:
    """Empty callback to ensure that each command function is separate.

    https://typer.tiangolo.com/tutorial/commands/one-or-multiple/#one-command-and-one-callback
    """
    pass  # noqa: WPS420


@app.command("teach-edh")
def create_teach_edh_instances(
    teach_edh_instances_base_dir: Path = settings.paths.teach_edh_instances,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for TEACh EDH Instances."""
    edh_instance_dir_paths = {
        DatasetSplit.train: list(teach_edh_instances_base_dir.joinpath("train").iterdir()),
        DatasetSplit.valid_seen: list(
            teach_edh_instances_base_dir.joinpath("valid_seen").iterdir()
        ),
        DatasetSplit.valid_unseen: list(
            teach_edh_instances_base_dir.joinpath("valid_unseen").iterdir()
        ),
    }

    DownstreamDbCreator.from_one_instance_per_json(
        dataset_name=DatasetName.teach,
        source_per_split=edh_instance_dir_paths,
        instance_model_type=TeachEdhInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("coco-captioning")
def create_coco_captioning_instances(
    train_ids_path: Path,
    dev_ids_path: Path,
    test_ids_path: Path,
    restval_ids_path: Path,
    is_example_id: bool,
    captions_train_path: Path = settings.paths.coco.joinpath(  # noqa: WPS404
        "captions_train2017.json"
    ),
    captions_val_path: Path = settings.paths.coco.joinpath(  # noqa: WPS404
        "captions_val2017.json"
    ),
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for COCO Instances."""
    train_annotations = read_json(captions_train_path)["annotations"]
    val_annotations = read_json(captions_val_path)["annotations"]

    all_ann = train_annotations + val_annotations
    all_ann = sorted(all_ann, key=lambda x: x["image_id"])

    if is_example_id:
        # convert example to image ids
        train_cap_ids = np.load(train_ids_path)
        train_image_ids = np.array(
            list({example["image_id"] for example in all_ann if example["id"] in train_cap_ids})
        )
        dev_cap_ids = np.load(dev_ids_path)
        dev_image_ids = np.array(
            list({example["image_id"] for example in all_ann if example["id"] in dev_cap_ids})
        )
        test_cap_ids = np.load(test_ids_path)
        test_image_ids = np.array(
            list({example["image_id"] for example in all_ann if example["id"] in test_cap_ids})
        )
        restval_cap_ids = np.load(restval_ids_path)
        restval_image_ids = np.array(
            list({example["image_id"] for example in all_ann if example["id"] in restval_cap_ids})
        )
    else:
        train_image_ids = np.load(train_ids_path)
        dev_image_ids = np.load(dev_ids_path)
        test_image_ids = np.load(test_ids_path)
        restval_image_ids = np.load(restval_ids_path)

    grouped_annotations: dict[int, dict[str, Union[str, list[str]]]] = {}  # noqa: WPS234
    groups = groupby(all_ann, key=lambda x: x["image_id"])
    for image_id, grouped_image_annotations in groups:
        image_annotations = list(grouped_image_annotations)
        grouped_annotations[image_id] = {
            "image_id": str(image_id),
            "captions_id": [str(example["id"]) for example in image_annotations],
            "captions": [example["caption"] for example in image_annotations],
        }

    coco_captioning_splits: dict[  # noqa: WPS234
        DatasetSplit, list[dict[str, Union[str, list[str]]]]
    ] = {
        DatasetSplit.train: [
            ann for img_id, ann in grouped_annotations.items() if img_id in train_image_ids
        ],
        DatasetSplit.valid: [
            ann for img_id, ann in grouped_annotations.items() if img_id in dev_image_ids
        ],
        DatasetSplit.test: [
            ann for img_id, ann in grouped_annotations.items() if img_id in test_image_ids
        ],
        DatasetSplit.restval: [
            ann for img_id, ann in grouped_annotations.items() if img_id in restval_image_ids
        ],
    }

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.coco,
        source_per_split=coco_captioning_splits,
        instance_model_type=CocoInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("nlvr")
def create_nlvr_instances(
    nlvr_instances_base_dir: Path = settings.paths.nlvr,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for NLVR^2."""
    nlvr_dir_paths = {
        DatasetSplit.train: nlvr_instances_base_dir.joinpath("train.jsonl"),
        DatasetSplit.valid_seen: nlvr_instances_base_dir.joinpath("balanced_dev.jsonl"),
        DatasetSplit.valid_unseen: nlvr_instances_base_dir.joinpath("balanced_test1.jsonl"),
    }

    DownstreamDbCreator.from_jsonl(
        dataset_name=DatasetName.nlvr,
        source_per_split=nlvr_dir_paths,
        instance_model_type=NlvrInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("vqa-v2")
def create_vqa_v2_instances(
    vqa_v2_instances_base_dir: Path = settings.paths.vqa_v2,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for VQA-v2."""
    vqa_v2_dir_paths = get_vqa_v2_annotation_paths(vqa_v2_instances_base_dir)

    source_per_split = {}
    for split_paths in vqa_v2_dir_paths:
        source_per_split[split_paths.split] = load_vqa_v2_annotations(
            questions_path=split_paths.questions_path, answers_path=split_paths.answers_path
        )

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.vqa_v2,
        source_per_split=source_per_split,
        instance_model_type=VQAv2Instance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("winoground")
def create_winoground_instances(
    hf_auth_token: Optional[str] = typer.Option(  # noqa: WPS404
        None,
        envvar="HF_AUTH_TOKEN",
        help="Hugging Face authentication token. You can also specify this using the `HF_AUTH_TOKEN` environment variable.",
    ),
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Creates instances db for the Winoground benchmark."""
    DownstreamDbCreator.from_huggingface(
        huggingface_dataset_identifier="facebook/winoground",
        dataset_name=DatasetName.winoground,
        instance_model_type=WinogroundInstance,
        output_dir=output_dir,
        hf_auth_token=hf_auth_token,
    ).run(num_workers=num_workers)


@app.command("refcoco")
def create_refcoco_instances(
    refcoco_instances_base_dir: Path = settings.paths.refcoco,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for RefCOCOg (UMD)."""
    paths_per_split = load_refcoco_annotations(refcoco_instances_base_dir)

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.refcoco,
        source_per_split=paths_per_split,
        instance_model_type=RefCocoInstance,
        output_dir=output_dir,
    ).run(num_workers)


if __name__ == "__main__":
    app()
