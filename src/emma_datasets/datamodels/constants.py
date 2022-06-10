from enum import Enum, IntEnum

from emma_datasets.common.helpers import flip_list_map_elements


class AnnotationType(Enum):
    """Possible annotations available from a dataset.

    The values for these enums are used for automatically providing clear and consistent feedback
    to users. Ensure that any new annotations are formatted similarly to maintain consistency.
    """

    qa_pair = "QA Pair"
    caption = "Caption"
    region = "Region"
    scene_graph = "Scene Graph"
    action_trajectory = "Action Trajectory"
    task_description = "Task Description"


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
    nlvr = "NLVR^2"


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
    DatasetName.conceptual_captions: MediaType.image,
    DatasetName.sbu_captions: MediaType.image,
    DatasetName.teach: MediaType.video,
    DatasetName.nlvr: MediaType.video,
}

AnnotationDatasetMap: dict[AnnotationType, list[DatasetName]] = {
    AnnotationType.qa_pair: [DatasetName.gqa],
    AnnotationType.caption: [
        DatasetName.coco,
        DatasetName.epic_kitchens,
        DatasetName.alfred,
        DatasetName.conceptual_captions,
        DatasetName.sbu_captions,
        DatasetName.nlvr,
    ],
    AnnotationType.region: [DatasetName.visual_genome],
    AnnotationType.scene_graph: [DatasetName.gqa],
    AnnotationType.action_trajectory: [DatasetName.alfred, DatasetName.teach],
    AnnotationType.task_description: [DatasetName.alfred],
}


DatasetAnnotationMap = flip_list_map_elements(AnnotationDatasetMap)
