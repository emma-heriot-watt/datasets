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
        self.trajectories = self.temp.joinpath("trajectories/")

        # Dataset source files
        self.datasets = self.storage.joinpath("datasets/")

        self.coco = self.datasets.joinpath("coco/")
        self.visual_genome = self.datasets.joinpath("visual_genome/")
        self.gqa = self.datasets.joinpath("gqa/")
        self.epic_kitchens = self.datasets.joinpath("epic_kitchens/")
        self.alfred = self.datasets.joinpath("alfred/")
        self.teach = self.datasets.joinpath("teach/")

        self.coco_images = self.coco.joinpath("images/")
        self.visual_genome_images = self.visual_genome.joinpath("images/")
        self.gqa_images = self.gqa.joinpath("images/")
        self.gqa_questions = self.gqa.joinpath("questions/")
        self.gqa_scene_graphs = self.gqa.joinpath("scene_graphs/")
        self.epic_kitchens_frames = self.epic_kitchens.joinpath("frames/")
        self.alfred_data = self.alfred.joinpath("json_2.1.0/")

        # Databases for output
        self.databases = self.storage.joinpath("db/")

        self._create_dirs()

    def _create_dirs(self) -> None:  # noqa: WPS213
        self.temp.mkdir(parents=True, exist_ok=True)

        self.captions.mkdir(parents=True, exist_ok=True)
        self.qa_pairs.mkdir(parents=True, exist_ok=True)
        self.scene_graphs.mkdir(parents=True, exist_ok=True)
        self.regions.mkdir(parents=True, exist_ok=True)
        self.trajectories.mkdir(parents=True, exist_ok=True)

        self.datasets.mkdir(parents=True, exist_ok=True)

        self.coco.mkdir(parents=True, exist_ok=True)
        self.visual_genome.mkdir(parents=True, exist_ok=True)
        self.gqa.mkdir(parents=True, exist_ok=True)
        self.epic_kitchens.mkdir(parents=True, exist_ok=True)
        self.alfred.mkdir(parents=True, exist_ok=True)
        self.teach.mkdir(parents=True, exist_ok=True)

        self.coco_images.mkdir(parents=True, exist_ok=True)
        self.visual_genome_images.mkdir(parents=True, exist_ok=True)
        self.gqa_images.mkdir(parents=True, exist_ok=True)
        self.gqa_questions.mkdir(parents=True, exist_ok=True)
        self.gqa_scene_graphs.mkdir(parents=True, exist_ok=True)
        self.epic_kitchens_frames.mkdir(parents=True, exist_ok=True)
        self.alfred_data.mkdir(parents=True, exist_ok=True)

        self.databases.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Common settings class for use throughout the repository."""

    paths: Paths = Paths()
