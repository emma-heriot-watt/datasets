import itertools

from emma_datasets.datamodels.constants import (
    AnnotationDatasetMap,
    AnnotationType,
    DatasetModalityMap,
    DatasetName,
)


def test_all_dataset_names_are_included_in_modality_map() -> None:
    """Verify all the dataset names are included in `DatasetModalityMap`."""
    modality_map_names = list(DatasetModalityMap.keys())

    for dataset_name in DatasetName:
        assert dataset_name in modality_map_names


def test_all_dataset_names_are_included_in_annotation_dataset_map() -> None:
    """Verify all dataset names are included in `AnnotationDatasetMap`."""
    annotation_linked_datasets = set(itertools.chain.from_iterable(AnnotationDatasetMap.values()))

    for dataset_name in DatasetName:
        assert dataset_name in annotation_linked_datasets


def test_all_annotations_are_included_in_annotation_dataset_map() -> None:
    """Verify all annotations are included in `AnnotationDatasetMap`."""
    dataset_linked_annotations = list(AnnotationDatasetMap.keys())

    for annotation in AnnotationType:
        assert annotation in dataset_linked_annotations
