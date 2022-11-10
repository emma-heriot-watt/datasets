import os
from typing import Any

from emma_datasets.datamodels.datasets.utils.simbot_utils.data_augmentations import (
    AugmentationInstruction,
    SimBotObjectAttributes,
)
from emma_datasets.datamodels.datasets.utils.simbot_utils.paraphrasers import (
    GotoParaphraser,
    ToggleParaphraser,
)


class BaseActionCreator:
    """General action creator."""

    def __init__(self) -> None:
        self.action_type = "Base"
        self.paraphraser = self._no_paraphraser

    def __call__(self, augmentation_instruction: AugmentationInstruction) -> dict[str, Any]:
        """Create an instruction dict from an augmentation instruction."""
        object_id = augmentation_instruction.object_id
        attributes = augmentation_instruction.attributes

        image_name = self._flat_image_name(augmentation_instruction.image_name)
        colorimages = [image_name]
        mask = [augmentation_instruction.bbox]
        mission_id = f"{self.action_type}_{image_name}"

        synthetic_action = self._create_synthetic_action(object_id, mask, attributes, colorimages)
        instruction_dict = self._create_synthetic_instruction(
            object_id, attributes, mission_id, synthetic_action
        )
        return instruction_dict

    def _no_paraphraser(self, object_id: str, attributes: SimBotObjectAttributes) -> str:
        return f"{object_id}."

    def _flat_image_name(self, image_name: str) -> str:
        return "__".join(image_name.split(os.sep))

    def _create_synthetic_action(
        self,
        object_id: str,
        mask: list[list[int]],
        attributes: SimBotObjectAttributes,
        colorimages: list[str],
    ) -> dict[str, Any]:
        synthetic_action = {
            "id": 0,
            "type": self.action_type,
            self.action_type.lower(): {
                "object": {
                    "id": object_id,
                    "colorImageIndex": 0,
                    "mask": mask,
                    "attributes": attributes.dict(),
                },
            },
            "colorImages": colorimages,
            "final": True,
        }
        return synthetic_action

    def _create_synthetic_instruction(
        self,
        object_id: str,
        attributes: SimBotObjectAttributes,
        mission_id: str,
        synthetic_action: dict[str, Any],
    ) -> dict[str, Any]:
        synthetic_instruction = {
            "instruction": self.paraphraser(object_id, attributes),
            "actions": [0],
        }

        instruction_dict = {
            "instruction": synthetic_instruction,
            "actions": [synthetic_action],
            "mission_id": mission_id,
            "annotation_id": 0,
            "instruction_id": 0,
            "synthetic": True,
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
