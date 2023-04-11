import json
import logging
import os
import random
import shutil
import subprocess  # noqa: S404
from argparse import ArgumentParser
from typing import Any

import boto3
import torch
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from emma_datasets.common import get_progress
from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.utils.simbot_utils.high_level_key_processor import (
    HighLevelKeyProcessor,
)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SessionClient:
    """A simple client for retrieving sessions from the s3 bucket and dynamo db."""

    def __init__(
        self,
        primary_key: str = "session_id",
        resource_region: str = "us-east-1",
        table_name: str = "SIMBOT_MEMORY_TABLE",
    ) -> None:

        self._primary_key = primary_key
        self._resource_region = resource_region
        self._table_name = table_name

        self._db = boto3.resource("dynamodb", self._resource_region)
        self._table = self._db.Table(self._table_name)

    def get_all_session_turns_for_session(self, session_id: str) -> list[Any]:
        """Get all the turns for a given session."""
        try:
            response = self._table.query(
                KeyConditionExpression=Key(self._primary_key).eq(session_id)
            )
        except ClientError as err:
            error_code = err.response["Error"]["Code"]

            if error_code != "ConditionalCheckFailedException":
                logger.exception("Could not add turn to table.", exc_info=err)
                raise err
            return []

        parsed_responses = response["Items"]
        logger.debug(f"Successfully got previous {len(parsed_responses)} turns")
        return parsed_responses

    def download_from_s3(self, local_path: str, s3_url: str, is_folder: bool = False) -> None:
        """Download a file or folder from the s3 bucket."""
        if os.path.exists(local_path):
            logger.debug(f"{s3_url} has been download in {local_path}")
            return
        command = f"aws s3 cp {s3_url} {local_path}"
        if is_folder:
            command = f"{command} --recursive"
        logger.debug(f"Downloading {s3_url} into {local_path}")
        subprocess.call(  # noqa: S603
            command.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )


class CDFTrajectoryCreator:
    """Create trajectory data from CDF sessions."""

    def __init__(
        self,
        sessions_file: str,
        output_json: str,
        output_feature_directory: str,
        cache_path: str = "storage/datasets/simbot/trajectories_sessions",
        s3_sessions_bucket_url: str = "s3://emma-simbot-live-challenge/",
        s3_results_bucket_url: str = "s3://emma-simbot/results/simbot-trajectories/missions/",
        prefix_inclusion_probability: float = 0.2,
        paraphrases_per_template: int = 10,
    ):
        self.output_json = output_json
        self.output_feature_directory = output_feature_directory
        os.makedirs(self.output_feature_directory, exist_ok=True)
        self.cache_path = cache_path
        os.makedirs(self.cache_path, exist_ok=True)

        self._s3_sessions_bucket_url = s3_sessions_bucket_url
        self._s3_results_bucket_url = s3_results_bucket_url

        self._client = SessionClient()
        self._session_ids = self.read_all_sessions_from_file(sessions_file)

        self._action_set = {
            "goto",
            "pickup",
            "open",
            "close",
            "place",
            "pour",
            "toggle",
            "clean",
            "fill",
            "highlight",
        }
        self._act_intent = "<act><one_match>"

        arena_definitions = get_arena_definitions()
        assets_to_labels = arena_definitions["asset_to_label"]
        special_names = arena_definitions["special_asset_to_readable_name"]
        assets_to_labels.update(special_names)

        # We are losing information here since different assets can be mapped to the same label
        # Thats ok though, we just want to object label for paraphrasing later on
        self._labels_to_assets = {label: asset for asset, label in assets_to_labels.items()}

        self._high_level_key_processor = HighLevelKeyProcessor(
            prefix_inclusion_probability=prefix_inclusion_probability,
            paraphrases_per_template=paraphrases_per_template,
        )

    def should_skip_session_turn_for_trajectory(self, session_turn: dict[str, Any]) -> bool:
        """Skip turns that should not be added to the trajectory.

        These generally are: 1) Turns where the agent spoke, either to a confirmation or to any
        lightweight dialog. In practice this should never happen but nevertheless its useful to
        avoid noise in the data. 2) Turns that correspond to search routines, we generally dont
        want the policy model to handle this. 3) Turns where there is no interaction action.
        """
        interaction_intent_type = session_turn["intent"]["physical_interaction"]["type"]
        interaction_action = session_turn["actions"]["interaction"]
        return interaction_intent_type != self._act_intent or interaction_action is None

    def check_if_session_is_successful(self, session_id: str) -> bool:
        """Check if the agent completed the mission within the session."""
        session_result_json = f"{session_id}.json"
        local_path = os.path.join(self.cache_path, session_result_json)
        s3_url = os.path.join(self._s3_results_bucket_url, session_result_json)
        self._client.download_from_s3(local_path, s3_url, is_folder=False)

        # If for some reason there is no results json the session is invalid
        if not os.path.exists(local_path):
            return False

        with open(local_path) as fp:
            data = json.load(fp)

        challenge_goals = data["last_game_state"]["challengeProgress"]["ChallengeGoals"]
        all_challenges_completed = []
        for challenge_goal in challenge_goals:
            is_finished = challenge_goal["isFinished"]
            subgoals_finished = all(
                [subgoal["isFinished"] for subgoal in challenge_goal["subTasks"]]
            )
            if is_finished and not subgoals_finished:
                logger.warning(
                    f"{session_id} is supposed to be completed but there are incomplete subgoals"
                )
                challenge_completed = False
            elif is_finished:
                challenge_completed = True
            else:
                challenge_completed = False
            all_challenges_completed.append(challenge_completed)

        # A session is valid if all challenges are completed
        return all(all_challenges_completed)

    def create_trajectories(self) -> None:
        """Create trajectory annotations."""
        progress = get_progress()
        task_id = progress.add_task(
            "Creating trajectory annotations",
            visible=True,
            start=True,
            total=len(self._session_ids),
            comment="",
        )
        missions = {}
        with progress:
            for session_id in self._session_ids:
                is_valid_session = self.check_if_session_is_successful(session_id)
                if not is_valid_session:
                    progress.advance(task_id)
                    continue

                instruction = self.get_high_level_instruction(session_id)

                # This should be unique across all missions
                # The sessions_ids have the form T.DATE/MISSION-GOAL-RANDOM-STRING
                mission_id = session_id.replace("/", "__")

                # Get the session turns the session
                session_turns = self._client.get_all_session_turns_for_session(session_id)

                # Download all files from s3
                local_path = os.path.join(self.cache_path, session_id)
                s3_url = os.path.join(self._s3_sessions_bucket_url, session_id)
                self._client.download_from_s3(local_path, s3_url, is_folder=True)

                # Create the sequence of actions
                session_actions = self.create_actions_for_session(session_id, session_turns)

                missions[mission_id] = {
                    "human_annotations": [
                        {
                            "instructions": [
                                {
                                    "instruction": instruction,
                                    "actions": self._get_action_ids(session_actions),
                                }
                            ]
                        }
                    ],
                    "actions": session_actions,
                }

                progress.advance(task_id)

        with open(self.output_json, "w") as fp:
            json.dump(missions, fp, indent=4)

        shutil.rmtree(self.cache_path)

    def create_actions_for_session(
        self, session_id: str, session_turns: list[Any]
    ) -> list[dict[str, Any]]:
        """Create all actions for a trajectory."""
        actions = []
        action_id = 0
        for session_turn_dict in session_turns:
            session_turn = json.loads(session_turn_dict["turn"])
            should_skip_turn = self.should_skip_session_turn_for_trajectory(
                session_turn=session_turn
            )

            if should_skip_turn:
                continue

            interaction_action = session_turn["actions"]["interaction"]

            prediction_id = session_turn["prediction_request_id"]
            raw_model_output = interaction_action["raw_output"]

            decoded_action = self.parse_raw_model_output(raw_model_output)[0]

            action_metadata = self.get_metadata_from_raw_action(decoded_action)

            image_features_path = os.path.join(self.cache_path, session_id, f"{prediction_id}.pt")
            image_features = torch.load(image_features_path)

            frame_index = action_metadata["frame_index"] - 1
            object_index = action_metadata["object_index"] - 1

            frame_features = self._format_feature_dict(image_features[frame_index])

            torch.save(
                frame_features, os.path.join(self.output_feature_directory, f"{prediction_id}.pt")
            )

            # TODO: we should get the object asset here, possibly from the CDF json?
            entity = image_features[frame_index]["entity_labels"][object_index]
            object_asset = self._labels_to_assets[entity]

            action_dict = self.make_action(
                action_id=action_id,
                action_metadata=action_metadata,
                prediction_id=prediction_id,
                object_id=object_asset,
                mask=interaction_action[action_metadata["action_type"]]["object"]["mask"],
            )

            actions.append(action_dict)

            action_id += 1

        return actions

    def parse_raw_model_output(self, raw_model_output: str) -> list[str]:
        """Split the raw_model_output into a list of actions."""
        decoded_action_str = raw_model_output.replace("<s>", "")
        split_actions = decoded_action_str.split(".")
        actions = [action.strip() for action in split_actions if action]
        return actions

    def get_metadata_from_raw_action(self, raw_action: str) -> dict[str, Any]:
        """Deconstruct and parse the raw action."""
        # Remove the end of trajectory token from the action to process it. We
        # only care about it when figuring out the dialog.
        action_tokens = raw_action.replace(".", "").strip().split(" ")
        action_type, action_params = self.get_simbot_action_from_tokens(action_tokens)

        object_index, frame_index = self.get_actions_params_from_raw_action(action_params)
        action_metadata = {
            "action_type": action_type,
            "object_index": object_index,
            "frame_index": frame_index,
        }
        return action_metadata

    def get_actions_params_from_raw_action(  # noqa: WPS231
        self, action_params: list[str]
    ) -> tuple[int, int]:
        """Deconstruct and parse the raw action from the params."""
        object_index = None
        frame_index = 1

        for action_param in action_params:
            action_param = action_param.strip()

            if action_param.startswith("<vis_token") and action_param.endswith(">"):
                object_index = self.extract_index_from_special_token(action_param)

            elif action_param.startswith("<frame") and action_param.endswith(">"):
                frame_index = self.extract_index_from_special_token(action_param)

        if object_index is None:
            raise AssertionError("Found an action that has no visual token")

        return object_index, frame_index

    def get_simbot_action_from_tokens(self, action_tokens: list[str]) -> tuple[str, list[str]]:
        """Get the SimBot action from the decoded action string.

        Assumptions:
            - The action appears at the start of the `decoded_action_string`.
            - The action can be of a length more than 1.

        Example:
            - If decoded_action == `forward`, then return `Forward`
            - If decoded_action == `pickup mug`, then return `Pickup`
        """
        parsed_action_name = None
        action_name = None

        index = len(action_tokens)

        while index > 0:
            action_name = " ".join(action_tokens[:index])

            if action_name.lower() in self._action_set:
                parsed_action_name = action_name.lower()
                break

            index -= 1

        # If we don't have an action type, then we don't really know what to do at all.
        if parsed_action_name is None:
            raise AssertionError("The action name could not be parsed.")

        return (
            parsed_action_name,
            action_tokens[index:],
        )

    def read_all_sessions_from_file(self, sessions_file: str) -> list[str]:
        """Read all the input sessions."""
        with open(sessions_file) as fp:
            session_ids = [line.strip() for line in fp.readlines()]
        return session_ids

    def get_high_level_instruction(self, session_id: str) -> str:
        """Get the high level description from the session id."""
        # the session had has the following form: T.DATE/high-level-key
        high_level_key = self._high_level_key_processor(session_id.split("/")[1])

        paraphrases = high_level_key.paraphrases
        return random.choice(paraphrases)

    def extract_index_from_special_token(self, token: str) -> int:
        """Extract the token index from a special token."""
        return int(token.strip().split("_")[-1].replace(">", ""))

    def make_action(
        self,
        action_id: int,
        action_metadata: dict[str, Any],
        prediction_id: str,
        object_id: str,
        mask: list[list[int]],
    ) -> dict[str, Any]:
        """Make an action dictionary."""
        action_dict = {
            "id": action_id,
            "type": action_metadata["action_type"].capitalize(),
            "colorImages": [f"{prediction_id}.png"],
            action_metadata["action_type"]: {
                "object": {"id": object_id, "mask": mask, "colorImageIndex": 0}
            },
        }
        return action_dict

    def _get_action_ids(self, session_actions: list[dict[str, Any]]) -> list[int]:
        return [action["id"] for action in session_actions]

    def _format_feature_dict(
        self, frame_features: dict[str, torch.Tensor]
    ) -> dict[str, list[dict[str, torch.Tensor]]]:
        frame_features["bbox_features"] = frame_features["bbox_features"].cpu()
        frame_features["bbox_coords"] = frame_features["bbox_coords"].cpu()
        frame_features["bbox_probas"] = frame_features["bbox_probas"].cpu()
        frame_features["cnn_features"] = frame_features["cnn_features"].cpu()
        return {"frames": [{"features": frame_features}]}  # type: ignore[dict-item]


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--sessions_file",
        help="Path to input session CDFs",
        default="sessions.txt",
    )

    parser.add_argument(
        "--output_json",
        help="Path to output json",
        default="trajectories.json",
    )

    parser.add_argument(
        "--output_feature_directory",
        help="Path to output feature directory",
        default="trajectories_features",
    )

    args = parser.parse_args()

    CDFTrajectoryCreator(
        sessions_file=args.sessions_file,
        output_json=args.output_json,
        output_feature_directory=args.output_feature_directory,
    ).create_trajectories()
