from pathlib import Path

from emma_datasets.common.settings import Settings
from emma_datasets.datamodels.base_model import BaseInstance
from emma_datasets.datamodels.constants import MediaType


settings = Settings()


class NlvrInstance(BaseInstance):
    """The dataclass for an NLVR^2 instance."""

    label: str
    sentence: str
    synset: str
    left_url: str
    right_url: str
    identifier: str

    @property
    def image_ids(self) -> list[str]:
        """Generates the image identifiers from the global NLVR id.

        We assume a consistent naming of the image files associated with each example. Given the
        identifier `split-set_id-pair_id-sentence-id`, the left and right images are named `split-
        set_id-pair_id-img0.png` and `split-set_id-pair_id-img1.png` respectively.
        """
        split, set_id, pair_id, _ = self.identifier.split("-")
        return [f"{split}-{set_id}-{pair_id}-img{i}" for i in range(2)]

    @property
    def left_image_filename(self) -> str:
        """Returns the filename of the left image."""
        return f"{self.image_ids[0]}.png"

    @property
    def right_image_filename(self) -> str:
        """Returns the filename of the right image."""
        return f"{self.image_ids[1]}.png"

    @property
    def modality(self) -> MediaType:
        """Get the modality of the instance."""
        return MediaType.video

    @property
    def feature_id(self) -> str:
        """Derives the unique example id which does not include sentence_id."""
        split, set_id, pair_id, _ = self.identifier.split("-")

        return f"{split}-{set_id}-{pair_id}"

    @property
    def features_path(self) -> Path:
        """Get the path to the features for this instance.

        In this case we follow the convention reported in the following PR:
        https://github.com/emma-simbot/perception/pull/151

        Concretely this means the following:
        (train-9642-3-img0.png, train-9642-3-img1.png) -> train-9642-3.pt
        """
        return settings.paths.nlvr_features.joinpath(f"{self.feature_id}.pt")
