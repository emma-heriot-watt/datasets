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
    epic_kitchens = "Epic Kitchens"
    gqa = "GQA"
    visual_genome = "Visual Genome"
    teach = "TEACh"
    conceptual_captions = "Conceptual Captions"
    sbu_captions = "SBU Captions"
    nlvr = "NLVR^2"
    vqa_v2 = "VQA v2"
    ego4d = "Ego4D Benchmark Annotations"
    ego4d_nlq = "Ego4D Natural Language Queries"
    ego4d_moments = "Ego4D Moment Queries"
    ego4d_vq = "Ego4D Visual Queries"
    ego4d_narrations = "Ego4D Narrations"
    winoground = "Winoground"
    refcoco = "COCO Referring Expressions"
    simbot_missions = "Alexa Prize SimBot Mission data"
    simbot_instructions = "Alexa Prize SimBot Instruction data"
    simbot_actions = "Alexa Prize SimBot Action-level data"
    simbot_clarifications = "Alexa Prize SimBot Clarification data"
    simbot_planner = "Alexa Prize SimBot High-level Planner data"


class DatasetSplit(Enum):
    """Split type for the dataset."""

    train = "training"
    valid = "validation"
    test = "testing"
    valid_seen = "valid_seen"
    valid_unseen = "valid_unseen"
    test_seen = "test_seen"
    test_unseen = "test_unseen"
    restval = "rest_val"
    test_dev = "test_dev"


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
    DatasetName.vqa_v2: MediaType.image,
    DatasetName.ego4d: MediaType.video,
    DatasetName.ego4d_moments: MediaType.video,
    DatasetName.ego4d_narrations: MediaType.video,
    DatasetName.ego4d_nlq: MediaType.video,
    DatasetName.ego4d_vq: MediaType.video,
    DatasetName.winoground: MediaType.image,
    DatasetName.refcoco: MediaType.image,
    DatasetName.simbot_missions: MediaType.multicam,
    DatasetName.simbot_instructions: MediaType.multicam,
    DatasetName.simbot_actions: MediaType.multicam,
    DatasetName.simbot_clarifications: MediaType.multicam,
    DatasetName.simbot_planner: MediaType.multicam,
}

AnnotationDatasetMap: dict[AnnotationType, list[DatasetName]] = {
    AnnotationType.qa_pair: [
        DatasetName.gqa,
        DatasetName.coco,
        DatasetName.vqa_v2,
        DatasetName.ego4d_vq,
        DatasetName.ego4d_moments,
        DatasetName.ego4d_nlq,
        DatasetName.simbot_missions,
        DatasetName.simbot_instructions,
        DatasetName.simbot_actions,
        DatasetName.simbot_clarifications,
    ],
    AnnotationType.caption: [
        DatasetName.coco,
        DatasetName.epic_kitchens,
        DatasetName.alfred,
        DatasetName.conceptual_captions,
        DatasetName.sbu_captions,
        DatasetName.nlvr,
        DatasetName.ego4d,
        DatasetName.ego4d_narrations,
        DatasetName.winoground,
    ],
    AnnotationType.region: [DatasetName.visual_genome, DatasetName.refcoco],
    AnnotationType.scene_graph: [DatasetName.gqa],
    AnnotationType.action_trajectory: [
        DatasetName.alfred,
        DatasetName.teach,
        DatasetName.simbot_missions,
        DatasetName.simbot_instructions,
        DatasetName.simbot_actions,
        DatasetName.simbot_clarifications,
        DatasetName.simbot_planner,
    ],
    AnnotationType.task_description: [DatasetName.alfred],
}


DatasetAnnotationMap = flip_list_map_elements(AnnotationDatasetMap)
