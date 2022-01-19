from enum import Enum, IntEnum, auto


class Annotation(Enum):
    """Possible annotations available from a dataset."""

    qa_pair = auto()
    caption = auto()
    region = auto()
    scene_graph = auto()


class MediaType(IntEnum):
    """Types of media which can be stored from datasets."""

    # Image = R, G, B
    image = 3
    # Video = R, G, B, Time
    video = 4
    # Multicam = R, G, B, Time, Camera
    multicam = 5


class DatasetName(Enum):
    """The different datasets available."""

    alfred = "ALFRED"
    coco = "COCO"
    epic_kitchens = "Epic Kitchens"
    gqa = "GQA"
    open_images = "OpenImages"
    visual_genome = "Visual Genome"
    teach = "TEACh"


class DatasetSplit(Enum):
    """Split type for the dataset."""

    train = "training"
    valid = "validation"
    test = "testing"


DatasetModalityMap: dict[DatasetName, MediaType] = {
    DatasetName.coco: MediaType.image,
    DatasetName.gqa: MediaType.image,
    DatasetName.visual_genome: MediaType.image,
    DatasetName.epic_kitchens: MediaType.video,
    DatasetName.open_images: MediaType.image,
}

AnnotationDatasetMap: dict[Annotation, list[DatasetName]] = {
    Annotation.qa_pair: [DatasetName.gqa],
    Annotation.caption: [DatasetName.coco, DatasetName.epic_kitchens],
    Annotation.region: [DatasetName.visual_genome],
    Annotation.scene_graph: [DatasetName.gqa],
}
