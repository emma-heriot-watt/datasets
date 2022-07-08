import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.progress import BarColumn, Progress, TextColumn

from emma_datasets.common import Settings, use_rich_for_logging
from emma_datasets.common.progress import BatchesProcessedColumn, CustomTimeColumn
from emma_datasets.datamodels import DatasetName
from emma_datasets.io import extract_archive


use_rich_for_logging()
logger = logging.getLogger(__name__)

settings = Settings()
settings.paths.create_dirs()


class OrganiseDataset:
    """Organise archives from datasets to be extracted and moved if needed."""

    def __init__(self, dataset_path: Path, dataset_name: DatasetName) -> None:
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name

    def submit(
        self,
        description: str,
        file_names: list[str],
        pool: ThreadPoolExecutor,
        progress: Progress,
        output_dir: Optional[Path] = None,
        move_files_to_output_dir: bool = False,
    ) -> None:
        """Submit archives which need to be extracted into the same output directory."""
        archive_paths = [self._dataset_path.joinpath(archive) for archive in file_names]

        task_id = progress.add_task(
            description,
            start=False,
            visible=False,
            total=0,
            dataset_name=self._dataset_name.value,
            comment="",
        )

        for path in archive_paths:
            pool.submit(
                extract_archive,
                path=path,
                task_id=task_id,
                progress=progress,
                output_dir=output_dir,
                move_files_to_output_dir=move_files_to_output_dir,
            )


def organise_coco(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise the files from COCO."""
    organise_dataset = OrganiseDataset(settings.paths.coco, DatasetName.coco)

    organise_dataset.submit(
        description="Extracting metadata",
        file_names=["annotations_trainval2017.zip"],
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )
    organise_dataset.submit(
        description="Extracting images",
        file_names=["train2017.zip", "val2017.zip"],
        output_dir=settings.paths.coco_images,
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )


def organise_visual_genome(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise the files from Visual Genome."""
    organise_dataset = OrganiseDataset(settings.paths.visual_genome, DatasetName.visual_genome)

    organise_dataset.submit(
        description="Extracting metadata",
        file_names=["region_descriptions.json.zip", "image_data.json.zip"],
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )
    organise_dataset.submit(
        description="Extracting images",
        file_names=["images.zip", "images2.zip"],
        output_dir=settings.paths.visual_genome_images,
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )


def organise_gqa(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise the files from GQA."""
    organise_dataset = OrganiseDataset(settings.paths.gqa, DatasetName.gqa)
    organise_dataset.submit(
        description="Extracting questions",
        file_names=["questions1.2.zip"],
        output_dir=settings.paths.gqa.joinpath("questions/"),
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )
    organise_dataset.submit(
        description="Extracting scene graphs",
        file_names=["sceneGraphs.zip"],
        output_dir=settings.paths.gqa.joinpath("scene_graphs/"),
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )
    organise_dataset.submit(
        description="Extracting images",
        file_names=["images.zip"],
        output_dir=settings.paths.gqa_images,
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )


def organise_epic_kitchens(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise the files from EPIC-KITCHENS."""
    organise_dataset = OrganiseDataset(settings.paths.epic_kitchens, DatasetName.epic_kitchens)

    for tar_file in settings.paths.epic_kitchens.glob("*.tar"):
        organise_dataset.submit(
            description=f"Extracting RGB frames for {tar_file.stem}",
            file_names=[tar_file.name],
            output_dir=settings.paths.epic_kitchens_frames.joinpath(f"{tar_file.stem}/"),
            pool=pool,
            progress=progress,
            move_files_to_output_dir=True,
        )


def organise_alfred(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise files from ALFRED."""
    organise_dataset = OrganiseDataset(settings.paths.alfred, DatasetName.alfred)

    alfred_warning = """Raw data from ALFRED comes as one giant 7z file, which cannot be efficiently extracted using this CLI. This is a known issue and unfortunately you will need to extract this one file separately using your shell. Sorry about that.
    """
    logger.warning(alfred_warning)

    organise_dataset.submit(
        description="Extracting metadata",
        file_names=["json_2.1.0.7z"],
        output_dir=settings.paths.alfred,
        pool=pool,
        progress=progress,
    )
    # TODO(amit): This is going to take forever and needs to be better
    # organise_dataset.submit(
    #     description="Extracting images",
    #     file_names=["full_2.1.0.7z"],
    #     output_dir=settings.paths.alfred,
    #     pool=pool,
    #     progress=progress,
    # )


def organise_teach(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise the TEACh dataset."""
    organise_dataset = OrganiseDataset(settings.paths.teach, DatasetName.teach)

    organise_dataset.submit(
        description="Extracting all games",
        file_names=["all_games.tar.gz"],
        pool=pool,
        progress=progress,
    )

    organise_dataset.submit(
        description="Extracting experiment games",
        file_names=["experiment_games.tar.gz"],
        pool=pool,
        progress=progress,
    )

    organise_dataset.submit(
        description="Extracting EDH instances",
        file_names=["edh_instances.tar.gz"],
        pool=pool,
        progress=progress,
    )

    organise_dataset.submit(
        description="Extracting images and states",
        file_names=["images_and_states.tar.gz"],
        pool=pool,
        progress=progress,
    )


def organise_nlvr(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise the files from NLVR."""
    organise_dataset = OrganiseDataset(settings.paths.nlvr, DatasetName.nlvr)

    # NLVR^2 data are defined as JSONL. The file extension is JSON...
    for path in settings.paths.nlvr.iterdir():
        new_extension_path = Path(str(path).replace("json", "jsonl"))
        path.rename(new_extension_path)

    organise_dataset.submit(
        description="Extracting images",
        file_names=["train_img.zip", "dev_img.zip", "test1_img.zip"],
        output_dir=settings.paths.nlvr_images,
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )


def organise_vqa_v2(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise the annotation files from VQA-v2."""
    organise_dataset = OrganiseDataset(settings.paths.vqa_v2, DatasetName.vqa_v2)

    organise_dataset.submit(
        description="Extracting metadata",
        file_names=[
            "v2_Questions_Train_mscoco.zip",
            "v2_Questions_Val_mscoco.zip",
            "v2_Questions_Test_mscoco.zip",
            "v2_Annotations_Train_mscoco.zip",
            "v2_Annotations_Val_mscoco.zip",
        ],
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )

    organise_dataset.submit(
        description="Extracting images",
        file_names=["train2017.zip", "val2017.zip", "test2017.zip"],
        output_dir=settings.paths.vqa_v2_images,
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )


def organise_refcoco(pool: ThreadPoolExecutor, progress: Progress) -> None:
    """Extract and organise the annotation files from RefCOCOg."""
    organise_dataset = OrganiseDataset(settings.paths.refcoco, DatasetName.refcoco)

    organise_dataset.submit(
        description="Extracting metadata",
        file_names=[
            "refcocog.zip",
        ],
        pool=pool,
        progress=progress,
        move_files_to_output_dir=True,
    )


def organise_datasets(
    datasets: Optional[list[DatasetName]] = typer.Argument(  # noqa: WPS404
        None, case_sensitive=False, show_default=False
    ),
    num_threads: Optional[int] = typer.Option(  # noqa: WPS404
        None,
        help="Number of threads to use for parallel processing. This default to `min(32, os.cpu_count() + 4).",
    ),
) -> None:
    """Organise the datasets in the storage folder.

    This is going to take a while because of how many files there are, the size of them, and the
    fact that we're limited to using multithreading instead of multiprocessing.
    """
    switcher: dict[DatasetName, Callable[[ThreadPoolExecutor, Progress], None]] = {
        DatasetName.coco: organise_coco,
        DatasetName.visual_genome: organise_visual_genome,
        DatasetName.gqa: organise_gqa,
        DatasetName.epic_kitchens: organise_epic_kitchens,
        DatasetName.alfred: organise_alfred,
        DatasetName.teach: organise_teach,
        DatasetName.nlvr: organise_nlvr,
        DatasetName.vqa_v2: organise_vqa_v2,
        DatasetName.refcoco: organise_refcoco,
    }

    progress_bar = Progress(
        TextColumn("[white]{task.fields[dataset_name]}", justify="left"),
        TextColumn("[bold yellow][progress.description]{task.description}", justify="right"),
        BarColumn(),
        BatchesProcessedColumn(),
        CustomTimeColumn(),
        TextColumn("[bright_black i]{task.fields[comment]}[/]"),
    )

    if not datasets:
        progress_bar.console.log("No datasets provided, therefore organising all datasets...")
        datasets = list(DatasetName)

    with progress_bar:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            for dataset_name in datasets:
                switcher[dataset_name](pool, progress_bar)


if __name__ == "__main__":
    organise_datasets()
