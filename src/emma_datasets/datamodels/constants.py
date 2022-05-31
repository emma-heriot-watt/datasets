from enum import Enum, IntEnum, auto


class Annotation(Enum):
    """Possible annotations available from a dataset."""

    qa_pair = auto()
    caption = auto()
    region = auto()
    scene_graph = auto()
    action_trajectory = auto()


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
    epic_kitchens = "EPIC-KITCHENS"
    gqa = "GQA"
    visual_genome = "Visual Genome"
    teach = "TEACh"
    conceptual_captions = "Conceptual Captions"
    sbu_captions = "SBU Captions"


class DatasetSplit(Enum):
    """Split type for the dataset."""

    train = "training"
    valid = "validation"
    test = "testing"
    valid_seen = "valid_seen"
    valid_unseen = "valid_unseen"


DatasetModalityMap: dict[DatasetName, MediaType] = {
    DatasetName.coco: MediaType.image,
    DatasetName.gqa: MediaType.image,
    DatasetName.visual_genome: MediaType.image,
    DatasetName.epic_kitchens: MediaType.video,
    DatasetName.alfred: MediaType.video,
}

AnnotationDatasetMap: dict[Annotation, list[DatasetName]] = {
    Annotation.qa_pair: [DatasetName.gqa],
    Annotation.caption: [DatasetName.coco, DatasetName.epic_kitchens, DatasetName.alfred],
    Annotation.region: [DatasetName.visual_genome],
    Annotation.scene_graph: [DatasetName.gqa],
    Annotation.action_trajectory: [DatasetName.alfred],
}
