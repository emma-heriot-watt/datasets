from pathlib import Path
from typing import Union


def read_txt(path: Union[str, Path]) -> list[str]:
    """Read a txt file and return a list of strings."""
    with open(path) as fp:
        raw_lines = [line.strip() for line in fp.readlines()]

    return raw_lines
