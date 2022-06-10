from pathlib import Path

from emma_datasets.datamodels.datasets.nlvr import NlvrInstance


def test_can_load_nlvr_data(nlvr_instances_path: Path) -> None:
    assert nlvr_instances_path.exists()

    instances = []
    with open(nlvr_instances_path) as in_file:
        instances = [NlvrInstance.parse_raw(line) for line in in_file]

    assert instances, "The file doesn't contain any instances."

    parsed_instance = instances[0]

    assert parsed_instance
    assert parsed_instance.identifier
    assert len(parsed_instance.identifier.split("-")) == 4, "Identifier it's not correct."
