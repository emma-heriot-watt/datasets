from pathlib import Path
from typing import Optional

from rich_click import typer

from emma_datasets.common import Settings
from emma_datasets.datamodels import DatasetName, DatasetSplit
from emma_datasets.datamodels.datasets import TeachEdhInstance
from emma_datasets.datamodels.datasets.nlvr import NlvrInstance
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
        paths_per_split=edh_instance_dir_paths,
        instance_model_type=TeachEdhInstance,
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
        paths_per_split=nlvr_dir_paths,
        instance_model_type=NlvrInstance,
        output_dir=output_dir,
    ).run(num_workers)


if __name__ == "__main__":
    app()
