<!-- Adding a new downstream dataset -->

## Description

<!-- What have you added and why? -->

## Checklist

- [ ] I have read the ["How to add a new downstream dataset" instructions](docs/how-to-add-a-new-downstream-dataset.md)
- [ ] A subset of the raw data has been added for testing the new dataset
- [ ] This PRs metadata is complete

### Adding a new dataset

- [ ] The dataset name has been added to the `DatasetName` enum
- [ ] The value for the new option in `DatasetName` is formatted like the public dataset name
- [ ] The dataset name has been added to the `DatasetModalityMap`
- [ ] The dataset name has been included in the `AnnotationDatasetMap`

### Creating the instance

- [ ] A new instance exists for the dataset, and it inherits from `BaseInstance`
- [ ] The new instance model for the dataset has tests (in `tests/datamodels/`)
- [ ] The new instance has been included in `emma_datasets/datamodels/datasets/__init__.py`
- [ ] The class name for the instance is in pascal case
- [ ] The instance is well-documented to ensure others can easily know what is happening and why

### Adding the dataset to the `downstream` command

- [ ] There is a new command to process the downstream dataset
- [ ] The name of the command is consistent with other commands
- [ ] The function has a docstring for the help of the command
- [ ] The function is listed as a command
- [ ] There are tests for the new dataset with the `DownstreamDbCreator` (in `tests/test_downstream_db_creator.py`)
- [ ] There are tests for the new command (in `tests/commands/test_create_downstream_dbs.py`)
