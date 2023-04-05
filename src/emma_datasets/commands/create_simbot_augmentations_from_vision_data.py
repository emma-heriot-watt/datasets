import json
import math
import os
import shutil
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray
from rich.progress import Progress, TaskID
from torch.utils.data import DataLoader, IterableDataset

from emma_datasets.augmentations.simbot_augmentators import (
    BreakAugmentation,
    CleanAugmentation,
    FillPourAugmentation,
    GoToAugmentation,
    OpenCloseAugmentation,
    PickupAugmentation,
    PlaceAugmentation,
    ScanAugmentation,
    SearchAugmentation,
    ToggleAugmentation,
)
from emma_datasets.augmentations.simbot_augmentators.action_creators import (
    BaseActionCreator,
    BreakActionCreator,
    CleanActionCreator,
    CloseActionCreator,
    FillActionCreator,
    GotoActionCreator,
    OpenActionCreator,
    PickupActionCreator,
    PlaceActionCreator,
    PourActionCreator,
    ScanActionCreator,
    SearchActionCreator,
    ToggleActionCreator,
)
from emma_datasets.augmentations.simbot_augmentators.base_augmentator import BaseAugmentation
from emma_datasets.augmentations.simbot_augmentators.clip_image_diversity import CLIProcessor
from emma_datasets.common import Settings, get_progress
from emma_datasets.constants.simbot.simbot import get_class_thresholds, get_objects_asset_synonyms


settings = Settings()


class AugmentationVisionDataset(IterableDataset[dict[Any, Any]]):
    """Create additional trajectory data.

    Args:
        output_json_file: Output json file containing the augmentation dataset
        output_image_dir: Output directory containing the images for the augmentation dataset
        metadata_files: Path to metadata files used to create the augmentation dataset. These are read from the metadata_train.txt
        augmentations: A list of available object augmentations.
        action_creators: A list of available action creators. Each action creator is applied to to each object augmentation
        dataset_version: Should be > v4 as previous versions have different json format
        coordinate_margin: The minimum width, minimum height for a bounding box.
    """

    def __init__(
        self,
        output_json_file: Path,
        output_image_dir: Path,
        root_vision_path: Path,
        metadata_files: list[Path],
        vision_data_augmentations: dict[str, BaseAugmentation],
        action_creators: list[BaseActionCreator],
        coordinate_margin: float = 10,
    ) -> None:
        self.output_json_file = output_json_file
        self.output_image_dir = output_image_dir
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        self.root_vision_path = root_vision_path
        self.metadata_files = metadata_files
        self.vision_data_augmentations = vision_data_augmentations
        self.action_creators = {
            action_creator.action_type: action_creator for action_creator in action_creators
        }
        self._coordinate_margin = coordinate_margin
        self._class_thresholds = get_class_thresholds()
        self._start = 0
        self._end = len(self.metadata_files)
        self._cache = settings.paths.simbot.joinpath("augmentations")
        self._cache.mkdir(parents=True, exist_ok=True)

    def configure_worker_folders(self, num_workers: int) -> None:
        """Create the folder inside the cache for each worker."""
        if num_workers == 0:
            worker_folder = self._cache.joinpath("worker_0")
            worker_folder.mkdir(parents=True, exist_ok=True)
        else:
            for worker_id in range(num_workers):
                worker_folder = self._cache.joinpath(f"worker_{worker_id}")
                worker_folder.mkdir(parents=True, exist_ok=True)

    def gather(self) -> None:
        """Write the new annotations.

        Group the metadata in terms of action type and call the post-process for each action type.
        This is used in case where some augmentators require to do some post-processing after
        collecting the data from all workers, e.g downsampling.
        """
        worker_folders = list(self._cache.iterdir())
        metadata_per_action_type: dict[str, Any] = {}

        progress = get_progress()
        task_id = progress.add_task(
            "Gathering annotations",
            visible=True,
            start=True,
            total=len(worker_folders),
            comment="",
        )
        with progress:
            metadata_per_action_type = self._collect_metadata_from_workers(
                worker_folders, progress, task_id
            )

        progress = get_progress()
        task_id = progress.add_task(
            "Post-processing annotations",
            visible=True,
            start=True,
            total=len(metadata_per_action_type.keys()),
            comment="",
        )

        final_metadata = {}
        with progress:
            for action_type, annotations_per_action_type in metadata_per_action_type.items():
                final_metadata.update(
                    self.vision_data_augmentations[action_type].post_process_metadata(
                        annotations_per_action_type, self._class_thresholds
                    )
                )
                progress.advance(task_id)

        with open(self.output_json_file, "w") as out_file:
            json.dump(final_metadata, out_file, indent=4)

        progress = get_progress()
        task_id = progress.add_task(
            f"Writing images to {self.output_image_dir}",
            visible=True,
            start=True,
            total=len(final_metadata.keys()),
            comment="",
        )
        with progress:
            for _, action_metadata in final_metadata.items():
                progress.advance(task_id)
                image_name = action_metadata["actions"][0]["colorImages"][0]
                destination_color_image = Path(self.output_image_dir, image_name)
                if destination_color_image.exists():
                    continue
                source_color_image = Path(
                    self.root_vision_path, str(image_name).replace("__", os.sep)
                )
                shutil.copy(source_color_image, destination_color_image)

        shutil.rmtree(self._cache)

    def __len__(self) -> int:
        """Dataset len."""
        return len(self.metadata_files)

    def __iter__(self) -> Iterator[dict[Any, Any]]:
        """Iterate over dataset."""
        worker_info = torch.utils.data.get_worker_info()

        # single-process data loading, return the full iterator
        if worker_info is None:
            iter_start = self._start
            iter_end = self._end
            worker_output_folder = self._cache.joinpath("worker_0")
        # in a worker process, split the workload
        else:
            worker_id = worker_info.id
            per_worker = int(math.ceil((self._end - self._start) / float(worker_info.num_workers)))
            iter_start = self._start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self._end)
            worker_output_folder = self._cache.joinpath(f"worker_{worker_id}")

        for file_idx in range(iter_start, iter_end):
            metadata_json_path = self.metadata_files[file_idx]
            (annotations, room_name, robot_position) = self._load_metadata(metadata_json_path)
            augmentation_instructions = []
            for _, augmentation in self.vision_data_augmentations.items():
                full_image_name = metadata_json_path.parent.joinpath(
                    f"{metadata_json_path.stem.split('_')[0]}_color.png",
                )
                image_name = str(full_image_name.relative_to(self.root_vision_path))

                augmentation_instructions.extend(
                    augmentation(
                        annotations=annotations,
                        robot_position=robot_position,
                        image_name=image_name,
                        class_thresholds=self._class_thresholds,
                        room_name=room_name,
                    )
                )

            final_instructions = []
            for augmentation_instruction in augmentation_instructions:
                instruction_dict = self.action_creators[augmentation_instruction.action_type](
                    augmentation_instruction
                )
                final_instructions.append(instruction_dict)

            if final_instructions:
                json_file = str(metadata_json_path.relative_to(self.root_vision_path)).replace(
                    os.sep, "__"
                )
                worker_output_json = worker_output_folder.joinpath(json_file)
                self._write_to_cache(final_instructions, worker_output_json)
            yield {}

    def _load_metadata(
        self, metadata_json_path: Path
    ) -> tuple[dict[str, Any], str, NDArray[np.float32]]:
        metadata_json_full_path = self.root_vision_path.joinpath(metadata_json_path)

        with open(metadata_json_full_path) as fp:
            metadata = json.load(fp)

        image_annotations = metadata["image_annotations"]
        objects_annotations = metadata["response"]["objects"]

        image_annotations_dict = {
            image_ann["object_id"]: image_ann for image_ann in image_annotations
        }
        objects_annotations_dict = {
            objects_ann["objectID"]: objects_ann for objects_ann in objects_annotations
        }

        annotations = {}
        for object_id, image_ann in image_annotations_dict.items():
            is_valid = self._object_id_is_valid(
                object_id, image_annotations_dict, objects_annotations_dict
            )
            if not is_valid:
                continue

            bbox = image_ann["bbox"]
            (xmin, ymin, xmax, ymax) = bbox
            if (xmax - xmin) < self._coordinate_margin or (ymax - ymin) < self._coordinate_margin:
                continue

            annotations[object_id] = {
                "image_annotation": image_annotations_dict[object_id],
                "object_annotation": objects_annotations_dict[object_id],
            }

        room = metadata["cdf"]["scene"]["roomLocation"][0]
        robot_position = self._get_robot_position(objects_annotations_dict)

        return (annotations, room, robot_position)

    def _object_id_is_valid(
        self,
        object_id: str,
        image_annotations_dict: dict[str, Any],
        objects_annotations_dict: dict[str, Any],
    ) -> bool:
        if object_id == "Unassigned":
            return False

        if object_id not in image_annotations_dict:
            return False

        return object_id in objects_annotations_dict

    def _get_robot_position(self, objects_annotations_dict: dict[str, Any]) -> NDArray[np.float32]:
        # https://alexaprizesim-ldg5293.slack.com/archives/C02SQAFVDFY/p1666968772331429
        robot_position = np.array(
            [
                objects_annotations_dict["TAM_1"]["position"]["x"],
                objects_annotations_dict["TAM_1"]["position"]["y"],
                objects_annotations_dict["TAM_1"]["position"]["z"],
            ]
        )
        return robot_position

    def _write_to_cache(
        self, final_instructions: list[dict[str, Any]], worker_output_json: Path
    ) -> None:
        if worker_output_json.exists():
            with open(worker_output_json) as worker_in_file:
                metadata = json.load(worker_in_file)
        else:
            metadata = {}

        for instruction in final_instructions:
            metadata[instruction["mission_id"]] = instruction

        with open(worker_output_json, "w") as worker_out_file:
            json.dump(metadata, worker_out_file, indent=4)

    def _collect_metadata_from_workers(
        self, worker_folders: list[Path], progress: Progress, task_id: TaskID
    ) -> dict[str, Any]:
        metadata_per_action_type: dict[str, Any] = defaultdict(dict)
        for worker_folder in worker_folders:
            worker_json_files = worker_folder.iterdir()
            for worker_json_file in worker_json_files:
                with open(worker_json_file) as worker_file:
                    worker_metadata = json.load(worker_file)
                for key, annotation in worker_metadata.items():
                    action_type = annotation["actions"][0]["type"]
                    metadata_per_action_type[action_type][key] = annotation

            progress.advance(task_id)
        return metadata_per_action_type


def get_metadata_version(root_file_path: Union[str, Path]) -> str:
    """Get the version from a metadata filepath."""
    return str(root_file_path).split("object_detection_data_")[1][:2]


def load_all_metadata_files(
    root_vision_path: Path,
    metadata_file: Path,
    limit_examples: Optional[int] = None,
    dataset_version: Optional[str] = None,
) -> list[Path]:
    """Reads all the available image annotation files."""
    with open(metadata_file) as f:
        annotation_files = f.readlines()
    annotation_files = sorted([line.strip() for line in annotation_files])
    metadata_files_temp = sorted(
        [root_vision_path.joinpath(line.strip()) for line in annotation_files]
    )
    if dataset_version is not None:
        metadata_files_temp = [
            metadata_file
            for metadata_file in metadata_files_temp
            if get_metadata_version(metadata_file) == dataset_version
        ]

    if limit_examples is not None:
        metadata_files_temp = metadata_files_temp[:limit_examples]

    metadata_files = []

    progress = get_progress()
    task_id = progress.add_task(
        f"Loading metadata from file {metadata_file}",
        visible=True,
        start=True,
        total=len(metadata_files_temp),
        comment="",
    )
    with progress:
        for meta_path in metadata_files_temp:
            img_num = meta_path.name.split("_")[0]
            subroot_dir = meta_path.parent
            image_path = subroot_dir.joinpath(f"{img_num}_color.png")
            image_seg_path = subroot_dir.joinpath(f"{img_num}_seg.png")
            if image_path.exists() and image_seg_path.exists():
                metadata_files.append(Path(meta_path))
            progress.advance(task_id)

    return metadata_files


def collate_fn(batch: dict[str, Any]) -> dict[str, Any]:
    """Placeholder collate for dataloader."""
    return batch


def generate_data(dataset: AugmentationVisionDataset, num_workers: int = 0) -> None:
    """Iterate over the dataset."""
    data_generator = DataLoader(
        dataset, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x
    )

    progress = get_progress()
    task_id = progress.add_task(
        "Creating augmentation examples",
        visible=False,
        start=False,
        total=len(dataset),
        comment="",
    )
    progress.start_task(task_id)
    progress.update(task_id, visible=True)

    with progress:
        for _ in data_generator:
            progress.advance(task_id)

    dataset.gather()


def string_to_class(class_str: str) -> BaseAugmentation:
    """Switcher for augmentation classes."""
    switcher = {
        "Break": BreakAugmentation,
        "Clean": CleanAugmentation,
        "Pour": FillPourAugmentation,
        "Pickup": PickupAugmentation,
        "Place": PlaceAugmentation,
        "Fill": FillPourAugmentation,
        "Close": OpenCloseAugmentation,
        "Goto": GoToAugmentation,
        "Open": OpenCloseAugmentation,
        "Scan": ScanAugmentation,
        "Search": SearchAugmentation,
        "Toggle": ToggleAugmentation,
    }
    return switcher[class_str]  # type: ignore[return-value]


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--root_vision_path",
        type=Path,
        help="Path to the root directory containing the vision datasets",
        default=Path("/home/ubuntu/data/object_detection"),
    )

    parser.add_argument(
        "--report_path",
        type=Path,
        help="Path to the output report csv file",
    )

    parser.add_argument(
        "--input_metadata_txt_path",
        type=Path,
        help="Path to the root directory containing the vision datasets",
        default=Path(
            "/home/ubuntu/data/datav2_collapsev4_isvalidv4_rgv1.12_classfiltered_train_09_09_2022/metadata_train.txt"
        ),
    )

    parser.add_argument(
        "--output_json_file",
        type=Path,
        help="Path to output json file",
        default=settings.paths.simbot.joinpath("train_augmentation_instructions.json"),
    )

    parser.add_argument(
        "--output_image_dir",
        type=Path,
        help="Path to output image directory",
        default=settings.paths.simbot.joinpath("train_augmentation_images"),
    )

    parser.add_argument(
        "--limit_examples",
        type=int,
        help="Limit of examples",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        help="Use only examples from a specific dataset version",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers",
    )

    parser.add_argument(
        "--augmentation_config",
        default="src/emma_datasets/constants/simbot/augmentations.json",
        help="Path to augmentation config",
    )

    args = parser.parse_args()

    root_vision_path = args.root_vision_path
    report_path = args.report_path
    input_metadata_txt_path = args.input_metadata_txt_path

    metadata_files = load_all_metadata_files(
        root_vision_path=root_vision_path,
        metadata_file=input_metadata_txt_path,
        limit_examples=args.limit_examples,
        dataset_version=args.dataset_version,
    )

    object_synonyms = get_objects_asset_synonyms()
    action_creators = [
        BreakActionCreator(object_synonyms),
        CleanActionCreator(object_synonyms),
        CloseActionCreator(object_synonyms),
        GotoActionCreator(object_synonyms),
        FillActionCreator(object_synonyms),
        PlaceActionCreator(object_synonyms),
        PickupActionCreator(object_synonyms),
        PourActionCreator(object_synonyms),
        OpenActionCreator(object_synonyms),
        ScanActionCreator(object_synonyms),
        SearchActionCreator(object_synonyms),
        ToggleActionCreator(object_synonyms),
    ]
    vision_data_augmentations: dict[str, BaseAugmentation] = {}
    with open(args.augmentation_config) as fp:
        augmentation_config = json.load(fp)

    diverse_image_selector = CLIProcessor()
    for augmentation, augmentation_dict in augmentation_config.items():
        class_name = list(augmentation_dict.keys())[0]
        augmentation_class = string_to_class(augmentation)
        vision_data_augmentations[augmentation] = augmentation_class.from_yaml_config(
            **augmentation_dict[class_name],
            root_vision_path=root_vision_path,
            report_path=report_path,
            diverse_image_selector=diverse_image_selector,
        )

    dataset = AugmentationVisionDataset(
        output_json_file=args.output_json_file,
        output_image_dir=args.output_image_dir,
        root_vision_path=root_vision_path,
        metadata_files=metadata_files,
        vision_data_augmentations=vision_data_augmentations,
        action_creators=action_creators,
        coordinate_margin=10,
    )

    dataset.configure_worker_folders(args.num_workers)

    generate_data(dataset, num_workers=args.num_workers)
