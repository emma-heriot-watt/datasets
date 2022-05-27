import logging
import tarfile
from pathlib import Path
from typing import Iterator, Optional, TypeVar
from zipfile import ZipFile, ZipInfo

from py7zr import SevenZipFile
from rich.progress import Progress, TaskID


logger = logging.getLogger(__name__)

T = TypeVar("T", tarfile.TarInfo, ZipInfo)


class ExtractArchive:
    """Function to extract files from the archive.

    Grouped all the various methods together under this class because otherwise that makes the file
    very messy.
    """

    def __call__(
        self,
        path: Path,
        task_id: TaskID,
        progress: Progress,
        output_dir: Optional[Path] = None,
        move_files_to_output_dir: bool = False,
    ) -> None:
        """Extract all files from the provided archive.

        Args:
            path (Path): Path to the archive file.
            task_id (TaskID): Task ID for the progress bar.
            progress (Progress): An instance of a Rich progress bar.
            output_dir (Path, optional): Output directory for the files extracted. Defaults to the
                parent of the archive file if not specified.
            move_files_to_output_dir (bool): Whether to move files to the output
                directory, therefore removing any folder structure. Defaults to False.
        """
        self._verify_path_exists(path)

        output_dir = output_dir if output_dir is not None else path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        if path.name.endswith(".zip"):
            self.extract_from_zip(path, output_dir, task_id, progress, move_files_to_output_dir)

        if path.name.endswith(".tar") or path.name.endswith(".tar.gz"):
            self.extract_from_tar(path, output_dir, task_id, progress, move_files_to_output_dir)

        if path.name.endswith(".7z"):
            self.extract_from_7z(path, output_dir, task_id, progress)

        progress.update(task_id, visible=False)
        progress.console.log(f"Extracted {path.name}")

    def extract_from_zip(
        self,
        path: Path,
        output_dir: Path,
        task_id: TaskID,
        progress: Progress,
        move_files_to_output_dir: bool,
    ) -> None:
        """Extract all files from within a zip archive."""
        with ZipFile(path) as archive_file:
            progress.update(task_id, visible=True, comment="Getting file list")

            all_files = [
                zipped_file for zipped_file in archive_file.infolist() if not zipped_file.is_dir()
            ]

            self._start_progress(progress, task_id, len(all_files))

            archive_file.extractall(
                output_dir,
                members=self.members_iterator(
                    all_files,
                    file_name_attr="filename",
                    is_dir_attr="is_dir",
                    output_dir=output_dir,
                    task_id=task_id,
                    progress=progress,
                    move_files_to_output_dir=move_files_to_output_dir,
                ),
            )

    def extract_from_tar(
        self,
        path: Path,
        output_dir: Path,
        task_id: TaskID,
        progress: Progress,
        move_files_to_output_dir: bool,
    ) -> None:
        """Extract all files from within a tar archive."""
        with tarfile.open(path) as tar_file:
            progress.update(task_id, visible=True, comment="Getting file list")

            all_files = tar_file.getmembers()

            self._start_progress(progress, task_id, len(all_files))

            tar_file.extractall(
                output_dir,
                members=self.members_iterator(
                    members=all_files,
                    file_name_attr="name",
                    is_dir_attr="isdir",
                    output_dir=output_dir,
                    task_id=task_id,
                    progress=progress,
                    move_files_to_output_dir=move_files_to_output_dir,
                ),
            )

    def extract_from_7z(  # noqa: WPS114
        self,
        path: Path,
        output_dir: Path,
        task_id: TaskID,
        progress: Progress,
    ) -> None:
        """Extract all files from within a 7z archive.

        Uses slightly different logic because the functionality is from a package and not core
        Python. Therefore, the same implementation within `members_iterator()` can't be directly
        used.
        """
        progress.update(task_id, visible=True, comment="Opening file")
        with SevenZipFile(path) as zip_file:
            progress.update(task_id, visible=True, comment="Getting file list")

            all_file_info = zip_file.list()

            all_file_info = (
                file_info for file_info in all_file_info if not file_info.is_directory
            )

            all_files = []

            progress.update(task_id, visible=True, comment="Filtering file list")

            for file_info in all_file_info:
                all_files.append(file_info.filename)
                progress.update(task_id, total=progress.tasks[task_id].total + 1)

            progress.start_task(task_id)

            for file_name, binary_file in zip_file.read(targets=all_files).items():
                progress.update(task_id, comment=f"Extracting {file_name}")

                file_path = Path(file_name)
                extracted_path = output_dir.joinpath(file_path.parent)
                extracted_path.mkdir(parents=True, exist_ok=True)

                with open(extracted_path.joinpath(file_path.name), "wb") as output_file:
                    output_file.write(binary_file.getbuffer())

                progress.advance(task_id)

    def members_iterator(
        self,
        members: list[T],
        file_name_attr: str,
        is_dir_attr: str,
        output_dir: Path,
        task_id: TaskID,
        progress: Progress,
        move_files_to_output_dir: bool,
    ) -> Iterator[T]:
        """Iterate through members of an archive, moving if needed and updating the progress."""
        for member in members:
            filename: str = getattr(member, file_name_attr)
            progress.update(task_id, comment=f"Extracting {filename}")

            yield member

            if move_files_to_output_dir:
                extracted_path = output_dir.joinpath(filename)

                if not getattr(member, is_dir_attr)() and extracted_path.parent != output_dir:
                    extracted_path.rename(output_dir.joinpath(extracted_path.name))

            progress.advance(task_id)

    def _start_progress(self, progress: Progress, task_id: TaskID, updated_total: int) -> None:
        progress.start_task(task_id)
        progress.update(
            task_id,
            visible=True,
            total=progress._tasks[task_id].total + updated_total,  # noqa: WPS437
        )

    def _verify_path_exists(self, path: Path) -> None:
        """Verify the file path exists, and warn if it doesn't."""
        if not path.exists():
            logger.warning(f"File [u]{path}[/] does not exist.")


extract_archive = ExtractArchive().__call__  # noqa: WPS609
