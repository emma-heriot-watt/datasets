import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Union


InputPathType = Union[Iterable[str], Iterable[Path], str, Path]
AnnotationPaths = Union[
    Iterable[str],
    Iterable[Path],
    str,
    Path,
    Iterable[tuple[Path, Path]],
]


def get_paths_from_dir(dir_path: Path) -> Iterator[Path]:
    """Get paths of all files from a directory."""
    if not dir_path.is_dir():
        raise RuntimeError("`dir_path` should point to a directory.")

    paths_from_dir = dir_path.rglob("*.*")
    files_from_dirs = (file_path for file_path in paths_from_dir if file_path.is_file())

    return files_from_dirs


def convert_strings_to_paths(string_paths: Iterable[str]) -> Iterable[Path]:
    """Convert strings to Path objects."""
    return [Path(path) for path in string_paths]


def _get_all_paths(paths: Iterable[Path]) -> list[Path]:
    non_dir_paths = (path for path in paths if path.is_file())
    dir_paths = (path for path in paths if path.is_dir())

    files_from_dirs = (get_paths_from_dir(dir_path) for dir_path in dir_paths)

    return list(itertools.chain(non_dir_paths, *files_from_dirs))


def get_all_file_paths(paths: AnnotationPaths) -> list[Path]:
    """Get all file paths from string paths."""
    if isinstance(paths, str):
        paths = convert_strings_to_paths([paths])

    if isinstance(paths, Path):
        paths = [paths]

    if isinstance(paths, Iterable) and all(isinstance(path, str) for path in paths):
        paths = convert_strings_to_paths(paths)  # type: ignore[arg-type]

    return _get_all_paths(paths)  # type: ignore[arg-type]
