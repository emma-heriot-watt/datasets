import itertools
from pathlib import Path
from typing import Iterable, Iterator, Union, overload


InputPathType = Union[Iterable[str], Iterable[Path], str, Path]


def get_paths_from_dir(dir_path: Path) -> Iterator[Path]:
    """Get paths of all files from a directory."""
    if not dir_path.is_dir():
        raise RuntimeError("`dir_path` should point to a directory.")

    paths_from_dir = dir_path.glob("**/*")
    files_from_dirs = (file_path for file_path in paths_from_dir if file_path.is_file())

    return files_from_dirs


def convert_strings_to_paths(string_paths: Iterable[str]) -> Iterable[Path]:
    """Convert strings to Path objects."""
    return [Path(path) for path in string_paths]


def _get_all_paths(paths: Iterable[Path]) -> Iterator[Path]:
    non_dir_paths = (path for path in paths if path.is_file())
    dir_paths = (path for path in paths if path.is_dir())

    files_from_dirs = (get_paths_from_dir(dir_path) for dir_path in dir_paths)

    return itertools.chain(non_dir_paths, *files_from_dirs)


@overload
def get_all_file_paths(paths: str) -> Iterator[Path]:
    converted_paths = convert_strings_to_paths([paths])
    return _get_all_paths(converted_paths)


@overload
def get_all_file_paths(paths: Iterable[str]) -> Iterator[Path]:
    converted_paths = convert_strings_to_paths(paths)
    return _get_all_paths(converted_paths)


@overload
def get_all_file_paths(paths: Path) -> Iterator[Path]:
    return _get_all_paths([paths])


@overload
def get_all_file_paths(paths: Iterable[Path]) -> Iterator[Path]:
    return _get_all_paths(paths)


def get_all_file_paths(paths: Union[Iterable[str], Iterable[Path], str, Path]) -> Iterator[Path]:
    """Get all file paths from string paths."""
    if isinstance(paths, str):
        paths = convert_strings_to_paths([paths])

    if isinstance(paths, Path):
        paths = [paths]

    if isinstance(paths, Iterable) and all(isinstance(path, str) for path in paths):
        paths = convert_strings_to_paths(paths)  # type: ignore[arg-type]

    return _get_all_paths(paths)  # type: ignore[arg-type]
