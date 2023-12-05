from pydantic import BaseModel, HttpUrl

from emma_datasets.datamodels.constants import DatasetSplit


class ConceptualCaptionsMetadata(BaseModel):
    """Represents the metadata of a Conceptual Caption example."""

    key: str
    caption: str
    url: HttpUrl
    width: int
    height: int
    shard_id: str
    dataset_split: DatasetSplit
