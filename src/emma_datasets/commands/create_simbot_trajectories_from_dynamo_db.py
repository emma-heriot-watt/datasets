import json
import logging
import math
import os
import random
import shutil
import subprocess  # noqa: S404
from argparse import ArgumentParser
from collections import Counter
from typing import Any, Literal, Optional

import boto3
import numpy as np
import pandas as pd
import torch
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from emma_datasets.common import get_progress
from emma_datasets.constants.simbot.simbot import get_arena_definitions
from emma_datasets.datamodels.datasets.utils.simbot_utils.high_level_key_processor import (
    HighLevelKey,
    HighLevelKeyProcessor,
)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CDFTrajectoryMetadata(BaseModel):
    """A basemodel for the wandb metadata for each CDF."""

    name: str = Field(alias="Name")
    state: str = Field(alias="State")
    notes: str = Field(alias="Notes")
    user: str = Field(alias="User")
    tags: Optional[str] = Field(alias="Tags")
    created: str = Field(alias="Created")
    runtime: int = Field(alias="Runtime")
    sweep: str = Field(alias="Sweep")
    cdf_floor_plan: int = Field(alias="cdf/floor_plan")
    cdf_layout: str = Field(alias="cdf/layout")
    cdf_room: str = Field(alias="cdf/room")
    cdf_scene_id: str = Field(alias="cdf/scene_id")
    high_level_key: str = Field(alias="high_level_key")
    high_level_key_action: str = Field(alias="high_level_key/action")
    high_level_key_converted_object: str = Field(alias="high_level_key/converted_object")
    high_level_key_from_receptacle: str = Field(alias="high_level_key/from_receptacle")
    high_level_key_from_receptacle_is_container: bool = Field(
        alias="high_level_key/from_receptacle_is_container"
    )
    high_level_key_interaction_object: str = Field(alias="high_level_key/interaction_object")
    high_level_key_target_object: str = Field(alias="high_level_key/target_object")
    high_level_key_target_object_color: str = Field(alias="high_level_key/target_object_color")
    high_level_key_to_receptacle: str = Field(alias="high_level_key/to_receptacle")
    high_level_key_to_receptacle_is_container: bool = Field(
        alias="high_level_key/to_receptacle_is_container"
    )
    preparation_session_id: str = Field(alias="preparation_session_id")
    session_id: str = Field(alias="session_id")
    is_success: int = Field(alias="is_success")
    subgoal_success_rate: float = Field(alias="subgoal_success_rate")

    @property
    def converted_high_level_key(self) -> str:
        """Converted high level key for each session."""
        return self.session_id

    @property
    def target_action_object(self) -> str:
        """Get the action_object pair.

        Used to make stratified splits.
        """
        return f"{self.high_level_key_action}_{self.high_level_key_target_object}"


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
        train_output_json: str,
        valid_output_json: str,
        output_feature_directory: str,
        upsample_factor: int = 1,
        cache_path: str = "storage/datasets/simbot/trajectories_sessions",
        s3_sessions_bucket_url: str = "s3://emma-simbot-live-challenge/",
        s3_results_bucket_url: str = "s3://emma-simbot/results/simbot-trajectories/missions/",
        prefix_inclusion_probability: float = 0.2,
        paraphrases_per_template: int = 10,
        split_valid_perc: float = 0.2,
        failed_sessions_txt: str = "failed_sessions.txt",
    ):
        self.output_json = {"train": train_output_json, "valid": valid_output_json}
        self.output_feature_directory = output_feature_directory
        os.makedirs(self.output_feature_directory, exist_ok=True)
        self.cache_path = cache_path
        os.makedirs(self.cache_path, exist_ok=True)

        self.upsample_factor = upsample_factor
        self.split_valid_perc = split_valid_perc
        self.failed_sessions_txt = failed_sessions_txt

        self._s3_sessions_bucket_url = s3_sessions_bucket_url
        self._s3_results_bucket_url = s3_results_bucket_url

        self._client = SessionClient()
        cdf_metadata = self.read_all_sessions_from_file(sessions_file)
        self._session_ids_per_split = self.split_sessions(cdf_metadata)

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
        self._search_intent = "<search>"
        self._goto_object_action_type = "goto object"

        arena_definitions = get_arena_definitions()
        assets_to_labels = arena_definitions["asset_to_label"]
        special_names = arena_definitions["special_asset_to_readable_name"]
        assets_to_labels.update(special_names)
        self._assets_to_labels = assets_to_labels

        # We are losing information here since different assets can be mapped to the same label
        self._labels_to_assets = {label: asset for asset, label in assets_to_labels.items()}

        # We need to add here anything that we want to preserve when doing the inverse map
        self._labels_to_assets["Apple"] = "Apple"

        self._high_level_key_processor = HighLevelKeyProcessor(
            prefix_inclusion_probability=prefix_inclusion_probability,
            paraphrases_per_template=paraphrases_per_template,
        )

    def split_sessions(
        self, cdf_metadata_list: list[CDFTrajectoryMetadata]
    ) -> dict[Literal["train", "valid"], list[CDFTrajectoryMetadata]]:
        """Split the sessions into train and validation."""
        all_objectives = np.array(
            [cdf_metadata.target_action_object for cdf_metadata in cdf_metadata_list]
        )

        objective_counts = Counter(all_objectives)
        train_sessions = []
        valid_sessions = []
        for objective, count in objective_counts.items():
            if count == 1:
                train_sessions.append(
                    cdf_metadata_list[np.where(all_objectives == objective)[0][0]]
                )
                continue

            objective_indices = np.where(all_objectives == objective)[0]
            random.shuffle(objective_indices)  # type: ignore[arg-type]

            valid_upper_bound = math.ceil(self.split_valid_perc * len(objective_indices))
            valid_indices = objective_indices[:valid_upper_bound]
            train_indices = objective_indices[valid_upper_bound:]

            train_sessions.extend([cdf_metadata_list[idx] for idx in train_indices])
            valid_sessions.extend([cdf_metadata_list[idx] for idx in valid_indices])

        return {"train": train_sessions, "valid": valid_sessions}

    def should_skip_session_turn_for_trajectory(  # noqa: WPS212, WPS231
        self,
        session_turn: dict[str, Any],
        previous_session_turn: Optional[dict[str, Any]] = None,
        is_last_turn: bool = False,
        add_goto_action_after_search: bool = True,
    ) -> bool:
        """Skip turns that should not be added to the trajectory.

        These generally are: 1) Turns where the agent spoke, either to a confirmation or to any
        lightweight dialog. In practice this should never happen but nevertheless its useful to
        avoid noise in the data. 2) Turns that correspond to search routines, we generally dont
        want the policy model to handle this. 3) Turns where there is no interaction action.
        """
        # Remove turns after a failed action
        environment = session_turn["intent"].get("environment", None)
        if environment is not None and environment["type"].startswith("<failure>"):
            return True

        actions = session_turn["actions"]
        interaction_action = actions.get("interaction", None)

        # Remove turns of failed actions
        failed_action = (
            interaction_action is not None
            and not is_last_turn  # last turn utterances dont have status
            and not interaction_action["status"]["success"]
        )
        if failed_action:
            return True

        interaction_intent_type = session_turn["intent"]["physical_interaction"]["type"]

        prev_environment = session_turn["intent"].get("environment", None)
        current_turn_is_after_error = prev_environment is not None and prev_environment[
            "type"
        ].startswith("<failure>")

        current_interaction_type = interaction_action["type"]
        # Skip any gotos that triggers act no match + search
        # We have already gone to the object from the find routine
        if previous_session_turn is not None and current_turn_is_after_error:
            previous_interaction_intent_type = previous_session_turn["intent"][
                "physical_interaction"
            ]["type"]
            previous_interaction_action = previous_session_turn["actions"]["interaction"]

            try:  # noqa: WPS229
                previous_entity = previous_interaction_action["goto"]["object"]["name"]
                current_entity = interaction_action["goto"]["object"]["name"]
                entity_condition = previous_entity == current_entity
            except KeyError:
                entity_condition = False

            should_skip_unecessary_goto = (
                previous_interaction_intent_type == self._search_intent
                and previous_interaction_action["type"] == self._goto_object_action_type
                and current_interaction_type == self._goto_object_action_type
                and entity_condition
            )
            if should_skip_unecessary_goto:
                return True

        # Add the last goto action after the search is complete + successful
        if add_goto_action_after_search:
            return (
                (interaction_intent_type != self._search_intent and session_turn["idx"] > 0)
                or interaction_action is None
                or current_interaction_type != self._goto_object_action_type
            )

        if session_turn["idx"] == 0 and not add_goto_action_after_search:
            return (
                interaction_action is None
                or interaction_intent_type != self._act_intent
                or current_interaction_type == self._goto_object_action_type
            )
        # Add only interaction actions, not from the search
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
                subgoal["isFinished"] for subgoal in challenge_goal["subTasks"]
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

    def run(self) -> None:
        """Create trajectory annotations."""
        for split in ("train", "valid"):
            self.create_trajectories_for_split(
                split=split,  # type: ignore[arg-type]
                sessions_for_split=self._session_ids_per_split[split],  # type:ignore[index]
            )

    def create_trajectories_for_split(  # noqa: WPS231
        self,
        split: Literal["train", "valid"],
        sessions_for_split: list[CDFTrajectoryMetadata],
    ) -> None:
        """Create trajectory annotations for train and validation split."""
        progress = get_progress()
        task_id = progress.add_task(
            f"Creating {split} trajectory annotations",
            visible=True,
            start=True,
            total=len(sessions_for_split),
            comment="",
        )
        missions = {}
        with progress:
            for session in sessions_for_split:
                logger.info(session)
                session_id = session.session_id

                is_valid_session = self.check_if_session_is_successful(session_id)
                high_level_key = self.process_highl_level_key(session_id=session_id)

                if not is_valid_session or high_level_key is None:
                    progress.advance(task_id)
                    continue

                # Get the session turns the session
                session_turns = self._client.get_all_session_turns_for_session(session_id)
                if not session_turns:
                    continue

                # Download all files from s3
                local_path = os.path.join(self.cache_path, session_id)
                s3_url = os.path.join(self._s3_sessions_bucket_url, session_id)
                self._client.download_from_s3(local_path, s3_url, is_folder=True)

                try:
                    missions_dict = self.create_mission_for_session(
                        high_level_key=high_level_key,
                        session_id=session_id,
                        session_turns=session_turns,
                    )
                except Exception:
                    logger.error(f"Could not create trajectory for {session_id}")

                missions.update(missions_dict)

                with open(self.output_json[split], "w") as fp:
                    json.dump(missions, fp, indent=4)

                progress.advance(task_id)

        with open(self.output_json[split], "w") as fp:  # noqa: WPS440
            json.dump(missions, fp, indent=4)

        shutil.rmtree(self.cache_path)

    def create_mission_for_session(
        self, high_level_key: HighLevelKey, session_id: str, session_turns: list[Any]
    ) -> dict[str, Any]:
        """Create all missions for a trajectory."""
        # This should be unique across all missions
        # The sessions_ids have the form T.DATE/MISSION-GOAL-RANDOM-STRING
        mission_id = session_id.replace("/", "__")

        agent_interacted_objects_assets = high_level_key.decoded_key.get_interacted_objects()
        instruction = random.choice(high_level_key.paraphrases)

        missions_dict = {}
        for add_goto in (True, False):
            session_actions = self.create_actions_for_session(
                session_id=session_id,
                session_turns=session_turns,
                agent_interacted_objects_assets=agent_interacted_objects_assets,
                add_goto_action_after_search=add_goto,
            )
            if not session_actions:
                continue

            # make sure that when add_goto = False the first action is not a goto
            if not add_goto:
                first_action = session_actions[0]
                first_action_type = first_action["type"]
                if first_action_type.lower() == "goto":
                    session_actions = session_actions[1:]

            missions_dict[f"{mission_id}_add_goto{add_goto}"] = {
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
        return missions_dict

    def create_actions_for_session(  # noqa: WPS210, WPS231
        self,
        session_id: str,
        session_turns: list[Any],
        agent_interacted_objects_assets: list[str],
        add_goto_action_after_search: bool = False,
    ) -> list[dict[str, Any]]:
        """Create all actions for a trajectory."""
        actions: list[dict[str, Any]] = []
        action_id = 0
        previous_session_turn = None
        for idx, session_turn_dict in enumerate(session_turns):
            session_turn = json.loads(session_turn_dict["turn"])
            should_skip_turn = self.should_skip_session_turn_for_trajectory(
                session_turn=session_turn,
                previous_session_turn=previous_session_turn,
                is_last_turn=idx == len(session_turns) - 1,
                add_goto_action_after_search=add_goto_action_after_search,
            )

            previous_session_turn = session_turn

            if should_skip_turn:
                continue

            if add_goto_action_after_search:
                add_goto_action_after_search = False

            interaction_action = session_turn["actions"]["interaction"]

            prediction_id = session_turn["prediction_request_id"]
            raw_model_output = interaction_action["raw_output"]

            decoded_action = self.parse_raw_model_output(raw_model_output)[0]

            action_metadata = self.get_metadata_from_raw_action(decoded_action)

            image_features_path = os.path.join(self.cache_path, session_id, f"{prediction_id}.pt")
            if not os.path.exists(image_features_path):
                logger.warning(
                    f"{image_features_path} does not exist, maybe the session is not on s3"
                )
                continue

            image_features = torch.load(image_features_path, map_location=torch.device("cpu"))
            frame_index = action_metadata["frame_index"] - 1
            object_index = action_metadata["object_index"] - 1
            frame_features = self._format_feature_dict(
                image_features[frame_index], image_name=f"{prediction_id}.png"
            )
            entity = image_features[frame_index]["entity_labels"][object_index]

            # Did the agent interacted with the object? If yes add in the trajectory data
            object_asset = self._agent_interacted_with_object_entity(
                entity, agent_interacted_objects_assets
            )

            if object_asset is None:
                object_asset = self._manually_fix_object_asset(
                    session_id=session_id,
                    prediction_id=prediction_id,
                    entity=entity,
                    agent_interacted_objects_assets=agent_interacted_objects_assets,
                )
                if object_asset is None:
                    return []

            if actions:
                previous_action = actions[-1]
                previous_action_type = previous_action["type"]

                if previous_action_type.lower() == action_metadata["action_type"] == "goto":
                    continue

            torch.save(
                frame_features,
                os.path.join(self.output_feature_directory, f"{prediction_id}.pt"),
            )

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

    def read_all_sessions_from_file(self, sessions_file: str) -> list[CDFTrajectoryMetadata]:
        """Read all the input sessions."""
        data_frame = pd.read_csv(sessions_file)

        successful_indices = data_frame["is_success"] > 0
        successful_sessions = data_frame[successful_indices]

        session_ids = []
        for _, row in successful_sessions.iterrows():
            cdf_metadata = CDFTrajectoryMetadata.parse_obj(row.to_dict())
            session_ids.append(cdf_metadata)

        failed_indices = data_frame["is_success"] == 0
        failed_sessions = data_frame[failed_indices]

        if failed_sessions.shape[0]:
            failed_session_names = []
            for _, failed_session in successful_sessions.iterrows():
                logger.warning(failed_session["Name"])
                failed_session_names.append(failed_session["Name"])

            with open(self.failed_sessions_txt, "w") as fp:
                for line in failed_session_names:
                    fp.write(f"{line}\n")

            logger.warning(f"Failed {failed_sessions.shape[0]}/{data_frame.shape[0]}")
        return session_ids

    def process_highl_level_key(self, session_id: str) -> HighLevelKey:
        """Get the high level description from the session id."""
        try:
            # the session had has the following form: T.DATE/high-level-key
            high_level_key = self._high_level_key_processor(session_id.split("/")[1])
        except Exception:
            logger.error(f"Could not convert the session id {session_id} to a high level key")
            return None  # type: ignore[return-value]

        return high_level_key

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
        self,
        frame_features: dict[str, torch.Tensor],
        image_name: str,
    ) -> dict[str, list[dict[str, torch.Tensor]]]:
        frame_features["bbox_features"] = frame_features["bbox_features"].cpu()
        frame_features["bbox_coords"] = frame_features["bbox_coords"].cpu()
        frame_features["bbox_probas"] = frame_features["bbox_probas"].cpu()
        frame_features["cnn_features"] = frame_features["cnn_features"].cpu()
        return {"frames": [{"features": frame_features, "image": image_name}]}  # type: ignore[dict-item]

    def _agent_interacted_with_object_entity(
        self, entity: str, agent_interacted_objects_assets: list[str]
    ) -> Optional[str]:
        if entity.lower() == "red button":
            return "ColorChanger_Button_Red"

        elif entity.lower() == "green button":
            return "ColorChanger_Button_Green"

        elif entity.lower() == "blue button":
            return "ColorChanger_Button_Blue"

        for object_asset, object_label in self._assets_to_labels.items():
            agent_interacted_with_object = (
                object_label.lower() == entity.lower()
                and object_asset in agent_interacted_objects_assets
            )
            if agent_interacted_with_object:
                return object_asset
        return None

    def _manually_fix_object_asset(  # noqa: WPS231
        self,
        session_id: str,
        prediction_id: str,
        entity: str,
        agent_interacted_objects_assets: list[str],
    ) -> Optional[str]:
        if entity.lower() == "bowl" and "FoodPlate_01" in agent_interacted_objects_assets:
            object_asset = "FoodPlate_01"
        elif entity.lower() == "plate" and "Bowl_01" in agent_interacted_objects_assets:
            object_asset = "Bowl_01"
        else:
            msg = f"I AM ABOUT TO SKIP session {session_id} because in {prediction_id} the {entity} was not found in {agent_interacted_objects_assets}"
            logger.warning(msg)
            while True:
                candidate_asset = input(  # noqa: WPS421
                    "Enter an object asset. Write none if you want to skip the session: "
                )
                if candidate_asset == "none":
                    logger.warning("Skipping the session")
                    return None
                elif candidate_asset not in self._assets_to_labels:
                    logger.warning(f"{candidate_asset} is not in the arena assets")
                else:
                    object_asset = candidate_asset
                    break
        return object_asset


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--sessions_file",
        help="Path to input session CDFs",
        default="sessions.txt",
    )

    parser.add_argument(
        "--train_output_json",
        help="Path to output json",
        default="train_trajectories.json",
    )

    parser.add_argument(
        "--valid_output_json",
        help="Path to output json",
        default="valid_trajectories.json",
    )

    parser.add_argument(
        "--output_feature_directory",
        help="Path to output feature directory",
        default="trajectories_features",
    )

    parser.add_argument(
        "--failed_sessions_txt",
        help="Path to output failed sessions txt file",
        default="failed_sessions.txt",
    )

    parser.add_argument(
        "--upsample_factor",
        help="Upsample factor for the trajectories",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    CDFTrajectoryCreator(
        sessions_file=args.sessions_file,
        train_output_json=args.train_output_json,
        valid_output_json=args.valid_output_json,
        output_feature_directory=args.output_feature_directory,
        upsample_factor=args.upsample_factor,
        failed_sessions_txt=args.failed_sessions_txt,
    ).run()
