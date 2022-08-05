from pathlib import Path
from typing import Union

import pandas as pd


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read a parquet file using pandas."""
    data = pd.read_parquet(path)
    return data
