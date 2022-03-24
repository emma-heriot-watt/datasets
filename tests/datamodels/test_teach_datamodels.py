import json
from pathlib import Path

from deepdiff import DeepDiff

from emma_datasets.datamodels.datasets.teach import (
    ExtendedTeachDriverAction,
    TeachEdhInstance,
    TeachInteraction,
)


def test_exported_parsed_edh_instance_is_identical_to_input(
    teach_edh_all_data_paths: list[Path],
) -> None:
    """Make sure that exported EDH instance from the datamodel is the same as the input.

    This is so that we can make sure that there are no issues when exporting for running inference.
    """
    for edh_instance_path in teach_edh_all_data_paths:
        assert edh_instance_path.exists()

        with edh_instance_path.open() as raw_instance_file:
            raw_instance = json.load(raw_instance_file)

        parsed_instance = TeachEdhInstance.parse_file(edh_instance_path).dict()

        comparison = DeepDiff(
            parsed_instance,
            raw_instance,
            ignore_numeric_type_changes=True,
            exclude_types=[type(None)],
        )

        # If the comparison dict is empty, then they are identical
        assert not comparison


def test_teach_edh_instance_interaction_has_custom_attributes(
    teach_edh_all_data_paths: list[Path],
) -> None:
    for edh_instance_path in teach_edh_all_data_paths:
        assert edh_instance_path.exists()

        parsed_instance = TeachEdhInstance.parse_file(edh_instance_path)

        for interaction in parsed_instance.interactions:
            assert isinstance(interaction.action_name, str)
            assert len(interaction.action_name)

            assert isinstance(interaction.frame_path, str)
            assert len(interaction.frame_path)

            assert isinstance(interaction.features_path, str)
            assert len(interaction.features_path)

            assert isinstance(interaction.agent_name, str)
            assert len(interaction.agent_name)

            if interaction.oid is not None:
                assert interaction.object_name


def test_teach_edh_instance_has_history_and_future_interactions(
    teach_edh_all_data_paths: list[Path],
) -> None:

    for edh_instance_path in teach_edh_all_data_paths:
        assert edh_instance_path.exists()

        instance = TeachEdhInstance.parse_file(edh_instance_path)

        for past_interaction in instance.interaction_history:
            assert isinstance(past_interaction, TeachInteraction)

        for future_interaction in instance.interactions_future:
            assert isinstance(future_interaction, TeachInteraction)


def test_teach_edh_instance_has_extended_driver_action_history(
    teach_edh_all_data_paths: list[Path],
) -> None:
    for edh_instance_path in teach_edh_all_data_paths:
        assert edh_instance_path.exists()

        instance = TeachEdhInstance.parse_file(edh_instance_path)

        assert instance.extended_driver_action_history

        for action in instance.extended_driver_action_history:
            assert isinstance(action, ExtendedTeachDriverAction)

            if action.action_id == 100:
                assert action.utterance

            if action.oid is not None:
                assert action.object_name
