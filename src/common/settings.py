from pathlib import Path

from pydantic import BaseSettings


BASE_DIR = Path("storage/")


class Paths:  # noqa: WPS230
    """Dataclass for data paths."""

    def __init__(self, base_dir: Path = BASE_DIR) -> None:
        self.storage = base_dir

        # Temp directories
        self.temp = self.storage.joinpath("temp/")

        self.captions = self.temp.joinpath("captions/")
        self.qa_pairs = self.temp.joinpath("qa_pairs/")
        self.scene_graphs = self.temp.joinpath("scene_graphs/")
        self.regions = self.temp.joinpath("regions/")
        self.scenes = self.temp.joinpath("scenes/")
        self.instances = self.temp.joinpath("instances/")

        # Dataset source files
        self.datasets = self.storage.joinpath("datasets/")

        self.coco = self.datasets.joinpath("coco/")
        self.visual_genome = self.datasets.joinpath("visual_genome/")
        self.gqa = self.datasets.joinpath("gqa/")

        # Databases for output
        self.databases = self.storage.joinpath("db/")

        self._create_dirs()

    def _create_dirs(self) -> None:
        self.temp.mkdir(parents=True, exist_ok=True)

        self.captions.mkdir(parents=True, exist_ok=True)
        self.qa_pairs.mkdir(parents=True, exist_ok=True)
        self.scene_graphs.mkdir(parents=True, exist_ok=True)
        self.regions.mkdir(parents=True, exist_ok=True)
        self.scenes.mkdir(parents=True, exist_ok=True)
        self.instances.mkdir(parents=True, exist_ok=True)

        self.datasets.mkdir(parents=True, exist_ok=True)

        self.coco.mkdir(parents=True, exist_ok=True)
        self.visual_genome.mkdir(parents=True, exist_ok=True)
        self.gqa.mkdir(parents=True, exist_ok=True)

        self.databases.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Common settings class for use throughout the repository."""

    paths: Paths = Paths()
