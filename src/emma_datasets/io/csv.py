import csv
from pathlib import Path
from typing import Union


def read_csv(path: Union[str, Path]) -> list[dict[str, str]]:
    """Read a CSV file and return a list of dictionaries.

    Because of how the DictReader works, every cell per row is just a string. If you want to
    convert it to a python object, you'll need to parse it or use `ast.literal_eval`. That is why
    the return type annotation is the way it is.
    """
    with open(path, encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    return data
