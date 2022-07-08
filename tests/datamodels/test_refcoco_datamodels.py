from pathlib import Path

from emma_datasets.datamodels.datasets.refcoco import RefCocoInstance, load_refcoco_annotations


def test_can_load_refcoco_data(refcoco_data_path: Path) -> None:
    raw_instances = load_refcoco_annotations(refcoco_data_path)
    assert len(raw_instances)


def image_metadata_of_refcoco_instance(instance: RefCocoInstance) -> None:
    # Image Metadata
    assert isinstance(instance.image_metadata.image_id, str)
    assert len(instance.image_metadata.image_id)
    assert isinstance(instance.image_metadata.width, int)
    assert instance.image_metadata.width > 0
    assert isinstance(instance.image_metadata.height, int)
    assert instance.image_metadata.height > 0


def region_of_refcoco_instance(instance: RefCocoInstance) -> None:
    # Region Metadata
    assert isinstance(instance.region.annotation_id, str)
    assert len(instance.region.annotation_id)
    assert isinstance(instance.region.x, float)
    assert instance.region.x >= 0
    assert isinstance(instance.region.y, float)
    assert instance.region.y >= 0
    assert isinstance(instance.region.w, float)
    assert instance.region.w > 0
    assert isinstance(instance.region.h, float)
    assert instance.region.h > 0


def referring_expression_of_refcoco_instance(instance: RefCocoInstance) -> None:
    # Referring Expression
    assert isinstance(instance.referring_expression.sentence, str)
    assert len(instance.referring_expression.sentence)
    assert isinstance(instance.referring_expression.sentence_id, str)
    assert len(instance.referring_expression.sentence_id)


def test_refcoco_data_has_custom_attributes(refcoco_data_path: Path) -> None:
    refcoco_annotations = load_refcoco_annotations(refcoco_data_path)
    for raw_instances in refcoco_annotations.values():
        for raw_instance in raw_instances:
            parsed_instance = RefCocoInstance.parse_obj(raw_instance)

            assert parsed_instance
            image_metadata_of_refcoco_instance(parsed_instance)
            region_of_refcoco_instance(parsed_instance)

            assert parsed_instance.image_metadata.image_id == parsed_instance.region.image_id

            referring_expression_of_refcoco_instance(parsed_instance)

            assert (
                parsed_instance.region.annotation_id
                == parsed_instance.referring_expression.annotation_id
            )
