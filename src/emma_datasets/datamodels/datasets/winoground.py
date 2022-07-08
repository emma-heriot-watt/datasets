from pathlib import Path
from typing import Union

from PIL.Image import Image

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import MediaType


class WinogroundInstance(BaseInstance):
    """A dataclass for the Winoground benchmark."""

    id: str
    image_0: Image  # noqa: WPS114
    image_1: Image  # noqa: WPS114
    caption_0: str  # noqa: WPS114
    caption_1: str  # noqa: WPS114
    tag: str
    secondary_tag: str
    num_main_preds: int
    collapsed_tag: str

    def modality(self) -> MediaType:
        """Returns the data modality for Winoground."""
        return MediaType.image

    def features_path(self) -> Union[Path, list[Path]]:
        """Returns the features path for Winoground images."""
        return Settings().paths.winoground_features.joinpath(f"{self.id}.pt")
