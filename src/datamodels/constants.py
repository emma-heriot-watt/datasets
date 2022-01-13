from enum import Enum, IntEnum


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
