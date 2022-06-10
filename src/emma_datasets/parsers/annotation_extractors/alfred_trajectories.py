import itertools
from typing import Any, Iterator

from overrides import overrides

from emma_datasets.datamodels import ActionTrajectory, AnnotationType, DatasetName
from emma_datasets.datamodels.datasets import AlfredMetadata
from emma_datasets.io import read_json
from emma_datasets.parsers.annotation_extractors.annotation_extractor import AnnotationExtractor


class AlfredTrajectoryExtractor(AnnotationExtractor[ActionTrajectory]):
    """Split subgoal trajectories for ALFRED into multiple files."""

    @property
    def annotation_type(self) -> AnnotationType:
        """The type of annotation extracted from the dataset."""
        return AnnotationType.action_trajectory

    @property
    def dataset_name(self) -> DatasetName:
        """The name of the dataset extracted."""
        return DatasetName.alfred

    def read(self, file_path: Any) -> dict[str, Any]:
        """Read ALFRED Metadata file."""
        return read_json(file_path)

    @overrides(check_signature=False)
    def convert(self, raw_feature: AlfredMetadata) -> list[tuple[int, ActionTrajectory]]:
        """Convert raw feature to a sequence of actions associated with a given subgoal.

        ALFRED uses a planner to generate a trajectory to complete a given high-level goal, and
        ALFRED has multiple language instructions for each trajectory. Therefore, it's possible
        that one language instruction can cover multiple subgoals, which can cause errors in how
        we've parsed and structured them.
        """
        trajectories: list[tuple[int, ActionTrajectory]] = []

        all_subgoals = itertools.groupby(raw_feature.plan.low_level_actions, lambda x: x.high_idx)
        for high_idx, subgoal in all_subgoals:
            trajectory = ActionTrajectory(
                low_level_actions=list(subgoal),
                high_level_actions=[raw_feature.plan.high_level_actions[high_idx]],
            )
            trajectories.append((high_idx, trajectory))

        num_language_instructions = min(
            len(ann.high_descs) for ann in raw_feature.turk_annotations["anns"]
        )
        num_high_level_subgoals = len(raw_feature.plan.high_level_actions)

        if num_high_level_subgoals != num_language_instructions:
            # Merge the last two subgoal actions so that they are aligned
            for act in trajectories[-1][1].low_level_actions:
                act.high_idx = num_language_instructions - 1

            new_final_trajectory = ActionTrajectory(
                # Merge the low level actions from the last two trajectories.
                low_level_actions=[
                    *trajectories[-2][1].low_level_actions,
                    *trajectories[-1][1].low_level_actions,
                ],
                high_level_actions=raw_feature.plan.high_level_actions[-2:],
            )

            # Delete the last two trrajectories
            trajectories = trajectories[:-2]

            # and add back the merged ones
            trajectories.append((num_language_instructions - 1, new_final_trajectory))

        return trajectories

    def process_single_instance(self, raw_instance: dict[str, Any]) -> None:
        """Process raw instance and write to file."""
        structured_instance = AlfredMetadata.parse_obj(raw_instance)
        trajectories = self.convert(structured_instance)
        self._process_subgoal_instances(
            task_id=structured_instance.task_id, trajectories=trajectories
        )
        self._process_trajectory_instance(
            task_id=structured_instance.task_id, trajectories=trajectories
        )

    def _process_subgoal_instances(
        self,
        task_id: str,
        trajectories: list[tuple[int, ActionTrajectory]],
    ) -> None:
        """Process subgoals."""
        for high_idx, trajectory in trajectories:
            traj_id = f"{task_id}_{high_idx}"
            self._write(trajectory, traj_id)

    def _process_trajectory_instance(
        self,
        task_id: str,
        trajectories: list[tuple[int, ActionTrajectory]],
    ) -> None:
        """Process full trajectory."""
        full_trajectory = self._merge_trajectories(trajectories)
        self._write(full_trajectory, f"{task_id}")

    def _merge_trajectories(
        self, trajectories: list[tuple[int, ActionTrajectory]]
    ) -> ActionTrajectory:
        """Merge trajectories for a full task."""
        high_indices = len(trajectories)
        traj_dict = dict(trajectories)
        low_level_actions = itertools.chain(
            *[traj_dict[high_idx].low_level_actions for high_idx in range(high_indices)]
        )

        high_level_actions = itertools.chain(
            *[traj_dict[high_idx].high_level_actions for high_idx in range(high_indices)]
        )

        return ActionTrajectory(
            low_level_actions=list(low_level_actions),
            high_level_actions=list(high_level_actions),
        )

    def _read(self) -> Iterator[Any]:
        """Reads all the trajectory metadata from the train and valid_seen data paths.

        For ALFRED we have to override this to make sure that all the single trajectory files are
        correctly combined in a single list.
        """
        raw_data = (
            self.process_raw_file_return(self.read(file_path)) for file_path in self.file_paths
        )

        return self.postprocess_raw_data(raw_data)
