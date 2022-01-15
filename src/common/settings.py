from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseSettings


DATA_DIR = Path("storage/data")


@dataclass
class Directories:
    """Dataclass for data directories."""

    def __init__(self, base_dir: Path = DATA_DIR) -> None:
        self.storage = base_dir
        self.captions = self.storage.joinpath("captions")
        self.qa_pairs = self.storage.joinpath("qa_pairs")
        self.scene_graphs = self.storage.joinpath("scene_graphs")
        self.regions = self.storage.joinpath("regions")
        self.databases = self.storage.joinpath("db")

        self.captions.mkdir(parents=True, exist_ok=True)
        self.qa_pairs.mkdir(parents=True, exist_ok=True)
        self.qa_pairs.mkdir(parents=True, exist_ok=True)
        self.scene_graphs.mkdir(parents=True, exist_ok=True)
        self.regions.mkdir(parents=True, exist_ok=True)
        self.databases.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Common settings class for use throughout the repository."""

    directories: Directories = Directories()
