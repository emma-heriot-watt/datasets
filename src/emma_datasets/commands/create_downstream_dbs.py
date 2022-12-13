from itertools import groupby
from pathlib import Path
from typing import Optional, Union

import numpy as np
from rich_click import typer

from emma_datasets.common import Settings
from emma_datasets.datamodels import DatasetName, DatasetSplit
from emma_datasets.datamodels.datasets import (
    CocoInstance,
    SimBotInstructionInstance,
    SimBotMissionInstance,
    TeachEdhInstance,
    VQAv2Instance,
    WinogroundInstance,
)
from emma_datasets.datamodels.datasets.ego4d import (
    Ego4DMomentsInstance,
    Ego4DNLQInstance,
    Ego4DVQInstance,
    load_ego4d_annotations,
)
from emma_datasets.datamodels.datasets.epic_kitchens import EpicKitchensInstance
from emma_datasets.datamodels.datasets.nlvr import NlvrInstance
from emma_datasets.datamodels.datasets.refcoco import RefCocoInstance, load_refcoco_annotations
from emma_datasets.datamodels.datasets.simbot import (
    load_simbot_action_annotations,
    load_simbot_annotations,
    load_simbot_clarification_annotations,
    load_simbot_planner_annotations,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    SimBotPlannerInstance,
)
from emma_datasets.datamodels.datasets.vqa_v2 import (
    get_vqa_v2_annotation_paths,
    load_vqa_v2_annotations,
    load_vqa_visual_genome_annotations,
    resplit_vqa_v2_annotations,
)
from emma_datasets.io import read_csv, read_json, read_txt
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
    divided_val_seen_path: Path = settings.paths.teach.joinpath(  # noqa: WPS404
        "divided_val_seen.txt"
    ),
    divided_val_unseen_path: Path = settings.paths.teach.joinpath(  # noqa: WPS404
        "divided_val_unseen.txt"
    ),
    divided_test_seen_path: Path = settings.paths.teach.joinpath(  # noqa: WPS404
        "divided_test_seen.txt"
    ),
    divided_test_unseen_path: Path = settings.paths.teach.joinpath(  # noqa: WPS404
        "divided_test_unseen.txt"
    ),
) -> None:
    """Create DB files for TEACh EDH Instances."""
    edh_instance_dir_paths = {
        DatasetSplit.train: list(teach_edh_instances_base_dir.joinpath("train").iterdir()),
        DatasetSplit.valid_seen: [
            teach_edh_instances_base_dir.joinpath("valid_seen", json_file)
            for json_file in read_txt(divided_val_seen_path)
        ],
        DatasetSplit.valid_unseen: [
            teach_edh_instances_base_dir.joinpath("valid_unseen", json_file)
            for json_file in read_txt(divided_val_unseen_path)
        ],
        DatasetSplit.test_seen: [
            teach_edh_instances_base_dir.joinpath("valid_seen", json_file)
            for json_file in read_txt(divided_test_seen_path)
        ],
        DatasetSplit.test_unseen: [
            teach_edh_instances_base_dir.joinpath("valid_unseen", json_file)
            for json_file in read_txt(divided_test_unseen_path)
        ],
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
    resplit_trainval: bool = False,
    include_visual_genome: bool = False,
) -> None:
    """Create DB files for VQA-v2."""
    vqa_v2_dir_paths = get_vqa_v2_annotation_paths(vqa_v2_instances_base_dir)

    source_per_split = {}
    for split_paths in vqa_v2_dir_paths:
        source_per_split[split_paths.split] = load_vqa_v2_annotations(
            questions_path=split_paths.questions_path, answers_path=split_paths.answers_path
        )
    if resplit_trainval:
        train_annotations, valid_annotations = resplit_vqa_v2_annotations(
            vqa_v2_instances_base_dir,
            train_annotations=source_per_split[DatasetSplit.train],
            valid_annotations=source_per_split[DatasetSplit.valid],
        )
        source_per_split[DatasetSplit.train] = train_annotations
        source_per_split[DatasetSplit.valid] = valid_annotations

    if include_visual_genome:
        source_per_split[DatasetSplit.train].extend(
            load_vqa_visual_genome_annotations(vqa_v2_instances_base_dir)
        )

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.vqa_v2,
        source_per_split=source_per_split,
        instance_model_type=VQAv2Instance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("ego4d_nlq")
def create_ego4d_nlq_instances(
    ego4d_nlq_instances_base_dir: Path = settings.paths.ego4d_annotations,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for Ego4D Natural Language queries."""
    ego4d_nlq_paths = {
        DatasetSplit.train: ego4d_nlq_instances_base_dir.joinpath("nlq_train.json"),
        DatasetSplit.valid: ego4d_nlq_instances_base_dir.joinpath("nlq_val.json"),
        DatasetSplit.test: ego4d_nlq_instances_base_dir.joinpath("nlq_test_unannotated.json"),
    }

    source_per_split = {}

    for split, split_path in ego4d_nlq_paths.items():
        source_per_split[split] = load_ego4d_annotations(split_path)

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.ego4d_nlq,
        source_per_split=source_per_split,
        instance_model_type=Ego4DNLQInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("ego4d_moments")
def create_ego4d_moments_instances(
    ego4d_moments_instances_base_dir: Path = settings.paths.ego4d_annotations,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for Ego4D vq queries."""
    ego4d_moments_paths = {
        DatasetSplit.train: ego4d_moments_instances_base_dir.joinpath("moments_train.json"),
        DatasetSplit.valid: ego4d_moments_instances_base_dir.joinpath("moments_val.json"),
        DatasetSplit.test: ego4d_moments_instances_base_dir.joinpath(
            "moments_test_unannotated.json"
        ),
    }

    source_per_split = {}

    for split, split_path in ego4d_moments_paths.items():
        source_per_split[split] = load_ego4d_annotations(split_path)

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.ego4d_vq,
        source_per_split=source_per_split,
        instance_model_type=Ego4DMomentsInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("ego4d_vq")
def create_ego4d_vq_instances(
    ego4d_vq_instances_base_dir: Path = settings.paths.ego4d_annotations,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for Ego4D Visual Queries."""
    ego4d_vq_paths = {
        DatasetSplit.train: ego4d_vq_instances_base_dir.joinpath("vq_train.json"),
        DatasetSplit.valid: ego4d_vq_instances_base_dir.joinpath("vq_val.json"),
        DatasetSplit.test: ego4d_vq_instances_base_dir.joinpath("vq_test_unannotated.json"),
    }

    source_per_split = {}

    for split, split_path in ego4d_vq_paths.items():
        source_per_split[split] = load_ego4d_annotations(split_path)

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.ego4d_vq,
        source_per_split=source_per_split,
        instance_model_type=Ego4DVQInstance,
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
    source_per_split = load_refcoco_annotations(refcoco_instances_base_dir)

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.refcoco,
        source_per_split=source_per_split,
        instance_model_type=RefCocoInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("simbot-missions")
def create_simbot_mission_instances(
    simbot_instances_base_dir: Path = settings.paths.simbot,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for Alexa Prize SimBot mission data."""
    source_per_split = load_simbot_annotations(simbot_instances_base_dir)

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.simbot_missions,
        source_per_split=source_per_split,
        instance_model_type=SimBotMissionInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("simbot-instructions")
def create_simbot_instruction_instances(
    simbot_instances_base_dir: Path = settings.paths.simbot,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
    train_num_additional_synthetic_instructions: int = 20000,
    valid_num_additional_synthetic_instructions: int = -1,
    add_synthetic_goto_instructions: bool = True,
) -> None:
    """Create DB files for Alexa Prize SimBot mission data."""
    source_per_split = load_simbot_annotations(
        simbot_instances_base_dir,
        annotation_type="instructions",
        train_num_additional_synthetic_instructions=train_num_additional_synthetic_instructions,
        valid_num_additional_synthetic_instructions=valid_num_additional_synthetic_instructions,
        add_synthetic_goto_instructions=add_synthetic_goto_instructions,
    )

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.simbot_instructions,
        source_per_split=source_per_split,
        instance_model_type=SimBotInstructionInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("simbot-actions")
def create_simbot_action_level_instances(
    simbot_instances_base_dir: Path = settings.paths.simbot,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for Alexa Prize SimBot mission data."""
    db_file_name = f"{DatasetName.simbot_instructions.name}"
    source_per_split = load_simbot_action_annotations(output_dir, db_file_name)
    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.simbot_actions,
        source_per_split=source_per_split,
        instance_model_type=SimBotInstructionInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("simbot-clarifications")
def create_simbot_clarification_instances(
    simbot_instances_base_dir: Path = settings.paths.simbot,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for Alexa Prize SimBot clarification data."""
    db_file_name = f"{DatasetName.simbot_instructions.name}"
    source_per_split = load_simbot_clarification_annotations(output_dir, db_file_name)
    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.simbot_clarifications,
        source_per_split=source_per_split,
        instance_model_type=SimBotInstructionInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("epic-kitchens")
def create_epic_kitchens_instances(
    epic_kitchens_instances_base_dir: Path = settings.paths.epic_kitchens,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for Epic-Kitchens."""
    epic_kitchens_paths = {
        DatasetSplit.train: epic_kitchens_instances_base_dir.joinpath("EPIC_100_train.csv"),
        DatasetSplit.valid: epic_kitchens_instances_base_dir.joinpath("EPIC_100_validation.csv"),
        DatasetSplit.test: epic_kitchens_instances_base_dir.joinpath(
            "EPIC_100_test_timestamps.csv"
        ),
    }

    source_per_split = {}

    for split, split_path in epic_kitchens_paths.items():
        split_annotations = read_csv(split_path)
        source_per_split[split] = split_annotations

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.epic_kitchens,
        source_per_split=source_per_split,
        instance_model_type=EpicKitchensInstance,
        output_dir=output_dir,
    ).run(num_workers)


@app.command("simbot-planner")
def create_simbot_high_level_planner_data(
    simbot_instances_base_dir: Path = settings.paths.simbot,
    alfred_data_dir: Path = settings.paths.alfred_data,
    output_dir: Path = settings.paths.databases,
    num_workers: Optional[int] = None,
) -> None:
    """Create DB files for Alexa Prize SimBot mission data."""
    source_per_split = load_simbot_planner_annotations(simbot_instances_base_dir, alfred_data_dir)

    DownstreamDbCreator.from_one_instance_per_dict(
        dataset_name=DatasetName.simbot_planner,
        source_per_split=source_per_split,
        instance_model_type=SimBotPlannerInstance,
        output_dir=output_dir,
    ).run(num_workers)


if __name__ == "__main__":
    app()
