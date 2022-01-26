from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from rich.progress import BarColumn, Progress, TextColumn

from src.common import Settings, use_rich_for_logging, use_rich_for_tracebacks
from src.common.progress import BatchesProcessedColumn, CustomTimeColumn
from src.datamodels import DatasetName
from src.io import extract_archive


use_rich_for_logging()
use_rich_for_tracebacks()
settings = Settings()


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


def organise_all_datasets() -> None:
    """Organise all the datasets in the storage folder.

    This is going to take a while because of how many files there are, the size of them, and the fact that we're limited to using multithreading instead of multiprocessing.

    If using an instance on a cloud server, ThreadPoolExecutor maxes out at 32 workers, so no need to have more cores than that.
    """
    progress_bar = Progress(
        TextColumn("[white]{task.fields[dataset_name]}", justify="left"),
        TextColumn("[bold yellow][progress.description]{task.description}", justify="right"),
        BarColumn(),
        BatchesProcessedColumn(),
        CustomTimeColumn(),
        TextColumn("[bright_black i]{task.fields[comment]}[/]"),
    )

    with progress_bar:
        with ThreadPoolExecutor() as pool:
            organise_coco(pool, progress_bar)
            organise_visual_genome(pool, progress_bar)
            organise_gqa(pool, progress_bar)
            organise_epic_kitchens(pool, progress_bar)
            organise_alfred(pool, progress_bar)
            organise_teach(pool, progress_bar)


if __name__ == "__main__":
    organise_all_datasets()
