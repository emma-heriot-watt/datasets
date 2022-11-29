import os
import random
from typing import Any, Union

from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    BaseParaphraser,
    GotoParaphraser,
    SearchParaphraser,
    ToggleParaphraser,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.simbot_datamodels import (
    AugmentationInstruction,
)


class BaseActionCreator:
    """General action creator."""

    def __init__(self) -> None:
        self.action_type = "Base"
        self.paraphraser: BaseParaphraser

    def __call__(self, augmentation_instruction: AugmentationInstruction) -> dict[str, Any]:
        """Create an instruction dict from an augmentation instruction."""
        synthetic_action = self._create_synthetic_action(augmentation_instruction)
        instruction_dict = self._create_synthetic_instruction(
            augmentation_instruction, synthetic_action
        )
        return instruction_dict

    def _create_mission_id(self, augmentation_instruction: AugmentationInstruction) -> str:
        image_name = self._flat_image_name(augmentation_instruction.image_name)
        return f"{self.action_type}_{augmentation_instruction.annotation_id}_{image_name}"

    def _flat_image_name(self, image_name: str) -> str:
        return "__".join(image_name.split(os.sep))

    def _create_synthetic_action(
        self, augmentation_instruction: AugmentationInstruction
    ) -> dict[str, Any]:

        attributes: Union[list[dict[str, Any]], dict[str, Any]]
        # This is currently only for the search action
        if isinstance(augmentation_instruction.attributes, list):
            attributes = [attribute.dict() for attribute in augmentation_instruction.attributes]
        else:
            attributes = augmentation_instruction.attributes.dict()

        image_name = self._flat_image_name(augmentation_instruction.image_name)
        colorimages = [image_name]

        synthetic_action = {
            "id": 0,
            "type": self.action_type,
            self.action_type.lower(): {
                "object": {
                    "id": augmentation_instruction.object_id,
                    "colorImageIndex": augmentation_instruction.image_index,
                    "mask": augmentation_instruction.bbox,
                    "attributes": attributes,
                },
            },
            "colorImages": colorimages,
            "final": True,
        }
        return synthetic_action

    def _create_synthetic_instruction(
        self,
        augmentation_instruction: AugmentationInstruction,
        synthetic_action: dict[str, Any],
    ) -> dict[str, Any]:

        # This is currently only for the search action
        if isinstance(augmentation_instruction.attributes, list):
            object_ids = augmentation_instruction.object_id
            search_object_initial_candidate_idx = random.randint(0, len(object_ids) - 1)
            object_id = object_ids[search_object_initial_candidate_idx]

            attributes = augmentation_instruction.attributes
            object_attributes = attributes[search_object_initial_candidate_idx]
        else:
            object_id = augmentation_instruction.object_id  # type: ignore[assignment]
            object_attributes = augmentation_instruction.attributes

        synthetic_instruction = {
            "instruction": self.paraphraser(object_id, object_attributes),
            "actions": [0],
        }
        mission_id = self._create_mission_id(augmentation_instruction)

        instruction_dict = {
            "instruction": synthetic_instruction,
            "actions": [synthetic_action],
            "mission_id": mission_id,
            "annotation_id": 0,
            "instruction_id": 0,
            "synthetic": True,
            "room_name": augmentation_instruction.room_name,
            "paraphrasable": True,
        }
        return instruction_dict


class ToggleActionCreator(BaseActionCreator):
    """Toggle action class."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        self.action_type = "Toggle"
        self.paraphraser = ToggleParaphraser(object_synonyms)


class GotoActionCreator(BaseActionCreator):
    """Goto action class."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        self.action_type = "Goto"
        self.paraphraser = GotoParaphraser(object_synonyms)


class SearchActionCreator(BaseActionCreator):
    """Search action class."""

    def __init__(self, object_synonyms: dict[str, list[str]]) -> None:
        self.action_type = "Search"
        self.paraphraser = SearchParaphraser(object_synonyms)

    def __call__(self, augmentation_instruction: AugmentationInstruction) -> dict[str, Any]:
        """Create the search instruction dictionary."""
        instruction_dict = super().__call__(augmentation_instruction=augmentation_instruction)
        instruction_dict["positive"] = augmentation_instruction.augmentation_metadata["positive"]  # type: ignore[index]
        return instruction_dict

    def _create_mission_id(self, augmentation_instruction: AugmentationInstruction) -> str:
        image_name = self._flat_image_name(augmentation_instruction.image_name)
        positive = augmentation_instruction.augmentation_metadata["positive"]  # type: ignore[index]
        return f"{self.action_type}_ispositive{positive}_{augmentation_instruction.annotation_id}_{image_name}"
