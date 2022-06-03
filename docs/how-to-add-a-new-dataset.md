# How to add a new dataset

Yay! New dataset!

## Updating the constants

Before anything, you need to add your new dataset to the constants because these are used throughout for keeping things consistent.

1. Add your dataset name to `DatasetName` (in `emma_datasets.datamodels.constants`) — ensure the value is in the proper format used by the dataset to keep things consistent and pretty
2. Include your new dataset name in `DatasetModalityMap` (in `emma_datasets.datamodels.constants`)
3. Based on the annotations your dataset has, add it to `AnnotationDatasetMap` (in `emma_datasets.datamodels.constants`)
