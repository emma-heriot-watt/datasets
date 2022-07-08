from pathlib import Path

from pydantic import BaseSettings

from emma_datasets.constants import constants_absolute_path


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
        self.task_descriptions = self.temp.joinpath("task_descriptions/")

        # Dataset source files
        self.datasets = self.storage.joinpath("datasets/")

        self.coco = self.datasets.joinpath("coco/")
        self.visual_genome = self.datasets.joinpath("visual_genome/")
        self.gqa = self.datasets.joinpath("gqa/")
        self.epic_kitchens = self.datasets.joinpath("epic_kitchens/")
        self.alfred = self.datasets.joinpath("alfred/")
        self.teach = self.datasets.joinpath("teach/")
        self.nlvr = self.datasets.joinpath("nlvr/")
        self.vqa_v2 = self.datasets.joinpath("vqa_v2/")
        self.winoground = self.datasets.joinpath("winoground/")
        self.refcoco = self.datasets.joinpath("refcoco/")

        self.coco_images = self.coco.joinpath("images/")
        self.visual_genome_images = self.visual_genome.joinpath("images/")
        self.gqa_images = self.gqa.joinpath("images/")
        self.gqa_questions = self.gqa.joinpath("questions/")
        self.gqa_scene_graphs = self.gqa.joinpath("scene_graphs/")
        self.epic_kitchens_frames = self.epic_kitchens.joinpath("frames/")
        self.alfred_data = self.alfred.joinpath("full_2.1.0/")
        self.teach_edh_instances = self.teach.joinpath("edh_instances/")
        self.nlvr_images = self.nlvr.joinpath("images/")
        self.vqa_v2_images = self.coco_images
        self.refcoco = self.coco_images

        self.coco_features = self.coco.joinpath("image_features/")
        self.visual_genome_features = self.visual_genome.joinpath("image_features/")
        self.gqa_features = self.gqa.joinpath("image_features/")
        self.epic_kitchens_features = self.epic_kitchens.joinpath("frame_features/")
        self.alfred_features = self.alfred.joinpath("frame_features/")
        self.teach_edh_features = self.teach.joinpath("frame_features/")
        self.nlvr_features = self.nlvr.joinpath("image_features/")
        self.vqa_v2_features = self.coco_features
        self.winoground_features = self.winoground.joinpath("image_features/")
        self.refcoco_features = self.coco_features

        # Databases for output
        self.databases = self.storage.joinpath("db/")

        self.constants = constants_absolute_path

    def create_dirs(self) -> None:  # noqa: WPS213
        """Create directories for files if they do not exist."""
        self.temp.mkdir(parents=True, exist_ok=True)

        self.captions.mkdir(parents=True, exist_ok=True)
        self.qa_pairs.mkdir(parents=True, exist_ok=True)
        self.scene_graphs.mkdir(parents=True, exist_ok=True)
        self.regions.mkdir(parents=True, exist_ok=True)
        self.trajectories.mkdir(parents=True, exist_ok=True)
        self.task_descriptions.mkdir(parents=True, exist_ok=True)

        self.datasets.mkdir(parents=True, exist_ok=True)

        self.coco.mkdir(parents=True, exist_ok=True)
        self.visual_genome.mkdir(parents=True, exist_ok=True)
        self.gqa.mkdir(parents=True, exist_ok=True)
        self.epic_kitchens.mkdir(parents=True, exist_ok=True)
        self.alfred.mkdir(parents=True, exist_ok=True)
        self.teach.mkdir(parents=True, exist_ok=True)
        self.nlvr.mkdir(parents=True, exist_ok=True)
        self.vqa_v2.mkdir(parents=True, exist_ok=True)
        self.winoground.mkdir(parents=True, exist_ok=True)
        self.refcoco.mkdir(parents=True, exist_ok=True)

        self.coco_images.mkdir(parents=True, exist_ok=True)
        self.visual_genome_images.mkdir(parents=True, exist_ok=True)
        self.gqa_images.mkdir(parents=True, exist_ok=True)
        self.gqa_questions.mkdir(parents=True, exist_ok=True)
        self.gqa_scene_graphs.mkdir(parents=True, exist_ok=True)
        self.epic_kitchens_frames.mkdir(parents=True, exist_ok=True)
        self.alfred_data.mkdir(parents=True, exist_ok=True)
        self.nlvr_images.mkdir(parents=True, exist_ok=True)

        self.databases.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Common settings class for use throughout the repository."""

    paths: Paths = Paths()
