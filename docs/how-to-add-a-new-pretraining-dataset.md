# How to add a new pretraining dataset

A **pretraining dataset** is specific used for pretraining the model and not for downstream fine-tuning or evaluating of the model. That means that each DB will **combine multiple datasets into one form**.

For model pretraining, we are making the assumption that we want to merge examples from multiple datasets into a single representation (an `Instance`).

At a high-level, these are the steps:

1. [Update the constants](#updating-the-constants)
2. [Create a subset of the full dataset for automated testing](#preparing-to-create-tests-for-the-new-dataset)
3. Processing a raw instance from the dataset
4. Extracting annotations from the dataset

## Things to note

### Pydantic

We use everything [Pydantic](https://pydantic-docs.helpmanual.io) has to offer: [Fields](https://pydantic-docs.helpmanual.io/usage/types/), [validators](https://pydantic-docs.helpmanual.io/usage/validators/), all of it!

### pytest

This guide assumes you have prior experience with [`pytest`](https://docs.pytest.org/) and unit-testing in general. To learn more about fantastic fixtures and how to use them, check out the following links:

- [About fixtures (from `pytest`)](https://docs.pytest.org/en/stable/explanation/fixtures.html#about-fixtures)
- [How to use fixtures (from
  `pytest`)](https://docs.pytest.org/en/stable/how-to/fixtures.html#how-to-fixtures)

### Typer

We also use [Typer](https://typer.tiangolo.com) for creating and testing the various CLI apps.

## Updating the constants

Before anything, you need to add your new dataset to the constants because these are used throughout for keeping things consistent.

1. Add your dataset name to `DatasetName` (in `emma_datasets.datamodels.constants`) — ensure the value is in the proper format used by the dataset to keep things consistent and pretty
2. Include your new dataset name in `DatasetModalityMap` (in `emma_datasets.datamodels.constants`)
3. Based on the annotations your dataset has, add it to `AnnotationDatasetMap` (in `emma_datasets.datamodels.constants`)

## Preparing to create tests for the new dataset

Datasets are huge and there might be edge cases you probably didn't consider. To reduce the likelihood of wasting time on manually checking the full pipeline for bugs, you can implement smaller fine-grained tests to verify and isolate any issues.

To prepare, you need to do the following:

1. Download a subset of instances and include the file(s) in `storage/fixtures/` — if you have multiple files, store them in a folder.
1. Create a fixture in `tests/fixtures/paths.py` which can point to the source data file/folder. Be sure to use `fixtures_root`.

That's it for now! We'll come back to using these files later in the guide.

## Creating models for the raw dataset

We need to easily parse every single instance and validate them to ensure that everything in the raw data is as you expect it! These classes are what get exported and stored within the DB file, and can be imported straight from the DB within your model.

Using [Pydantic's parsing methods](https://pydantic-docs.helpmanual.io/usage/models/#helper-functions), your raw data will be loaded from this file directly, and it will be validated to the schema of this model. This way, we can also verify that every single instance of the model is what you expect it to be, and you can deal with any weird data issues if they get flagged. If this happens, it will error.

The best example for what you are capable of doing is `TeachEdhInstance` (in `emma_datasets.datamodels.datasets.teach`). Each instance is represented by a single `TeachEdhInstance` and is used to import. It's made of multiple models, and Pydantic automatically parses all the nested models for you!

We advocate that the model for your dataset instance is as detailed as possible, using additional validators and properties where possible to ease usage downstream and ensure _every single instance_ is as expected.

### Creating an instance for your dataset

1. Create a new file for your dataset in `src/emma_datasets/datamodels/datasets/`.
1. Import `BaseModel` from `emma_datasets.datamodels.base_model`
1. Create a class for your instance, inheriting from `BaseInstance`. For consistency, please ensure that the name of your class is in pascal case.
1. Import your new instance into `emma_datasets/datamodels/datasets/__init__.py` to ensure it is easily accessible like the others.

_Note: We used Pascal case because we made some typos and we don't want to rename every single class throughout the library, so it's just staying that way for now._

### Modifying the data on load, automatically

If you need to consistently modify every single instance of the raw model, you can do that too! Use Pydantic's [`root_validator`](https://pydantic-docs.helpmanual.io/usage/validators/#root-validators) to take in the example and return what you want.

For examples, see [`AlfredMetadata`](https://github.com/emma-simbot/datasets/blob/4ea83c492cdab331ab7c722422f48ee8ee181659/src/emma_datasets/datamodels/datasets/alfred.py#L136-L144) and [`EpicKitchensNarrationMetdata`](https://github.com/emma-simbot/datasets/blob/4ea83c492cdab331ab7c722422f48ee8ee181659/src/emma_datasets/datamodels/datasets/epic_kitchens.py#L48-L56) to see how the raw data is modified on import.

#### What if the raw data is not in `snake_case`?

If the keys in the raw data are not in snake case, you can use the `alias` key in `Field`. Check out [the example here](https://pydantic-docs.helpmanual.io/usage/model_config/#alias-precedence) or in ALFRED-related models (in `src/emma_datasets/datamodels/datasets/alfred.py`)

### Testing your model works on the raw dataset

We want to verify that your data can be imported and parsed by your instance correctly. As shown by `tests/datamodels/test_teach_datamodels.py`, you can verify that your model is working on your raw data correctly without needing to run the entire pipeline.

Using your [previously created fixture paths](#preparing-to-create-tests-for-the-new-dataset), create a new module for your dataset within `tests/datamodels` and add your tests to it.

If you are unsure what tests to include, check out other tests within the same directory. If you include `@property`'s in your instance, then you'll also want to test them too as they should be present for all instances.

## Extracting annotations

For speed, we need to extract all the annotations from every instance of the dataset in advance, as a type of `Annotation` class. We use these `Annotation`s when we are creating the new instances. This way, it allows for easy importing of data that will also be validated, since `Annotation` inherits from Pydantic.

1. Create a new class that inherits from `AnnotationExtractor` within `src/emma_datasets/parsers/annotation_extractors/`. We recommend new files for each extractor.
2. Implement the `convert` and `process_single_instance` methods
3. Include your new class in `src/emma_datasets/parsers/annotation_extractors/__init__.py`. This way, it can be easily imported with the other annotation extractors.

Other methods are present to help with making the conversion process easier. We recommend checking out the documentation for each method in `src/emma_datasets/parsers/annotation_extractors/annotation_extractor.py`.

### Including the new extractor in the pipeline

A command exists in `emma_datasets.commands.extract_annotations` which extracts all the annotations from all the datasets. To ensure that your dataset is included in the pipeline, you need to include the extractor in this command.

To do this, you need to:

1. Create a new function that initializes and returns your specific annotation extractor.
2. Include this function in the `all_extractor_callables` list.

### Testing your annotation extractor

_NOTE: For this section, we assume you have [already created fixtures for the new dataset](#preparing-to-create-tests-for-the-new-dataset)._

To verify that your annotation extractor works and continues to do so, you need to implement some simple tests with a bit of prep work. For testing performance, we do not extract the annotations repeatedly. Therefore, we cache them to they are re-used in downstream tests. `pytest` allows us to do this automatically with minimum effort.

#### Create a new cached folder for the extracted annotations

1. Find the `extracted_annotations_paths()` fixture in`tests/fixtures/paths.py`
2. Within the `annotation_folders` variable, add a new key for your dataset's annotations. This will be unique to your dataset, so we recommend making it clear.

#### Creating a fixture for your annotation extractor

1. Go to `tests/fixtures/annotation_extractors.py`
2. Create a fixture that is similarly named to others in this module (generally following the pattern of `extract_<DATASET_NAME>_<ANNOTATION_TYPE>`)
3. The arguments to the fixture should be the fixture to the raw data paths [(as created above)](#preparing-to-create-tests-for-the-new-dataset), and `extracted_annotations_paths`. It should return a `bool`.
4. Like the other fixtures, call the `extracted_annotations()` function, providing it with the necessary arguments to run
5. Include your annotation extractor fixture in the `all_extracted_annotations()` fixture — updating both the function arguments and adding a new assert statement for it.

#### Creating the actual test for the annotation extractor

1. Create a test within `tests/test_annotation_extractors.py` for your annotation extractor, using your [previously created fixture](#creating-a-fixture-for-your-annotation-extractor)
2. Run the test to verify it is all working.

## Converting your dataset to a `DatasetMetadata`

Each instance is converted into a `DatasetMetadata` to be easily processed and merged into the single `Instance` format. The `DatasetMetadata` model provides a consistent way to store the metadata and access the annotations.

### Creating a dataset metadata parser

1. Create a new module for your metadata parser in `src/emma_datasets/parsers/dataset_metadata/`
2. Create your class, inheriting the base `DatasetMetadataParser` from `emma_datasets.parsers.dataset_metadata.metadata_parser`. `DatasetMetadataParser` is a generic class, meaning the model for the metadata of the raw dataset should be provided with it. For example, the `AlfredMetadataParser` inherits from `DatasetMetadataParser[AlfredMetadata]`
3. Update the class variables for `metadata_model` and `dataset_name`
4. Implement the private `_read()` method, which will read in the data for each path provided to `self.data_paths` within the `__init__()`
5. Implement the `convert_to_dataset_metadata` class method, to convert a single instance from the raw dataset into instances of `DatasetMetadata`

If you are unsure how this is implemented, use the other modules as examples.

#### When a single raw instance becomes multiple instances of `DatasetMetadata`

As was the case with ALFRED, each raw example is split by subgoals into multiple pretraining instances, and therefore needed to return multiple `DatasetMetadata`. Checkout how that was handled there.

## Aligning datasets: When datasets overlap

When you want to use multiple datasets but they overlap and come from common sources, it can be incredibly difficult to use them and handle these duplicates. This was the case for COCO, VisualGenome, GQA: VisualGenome and GQA used COCO, and GQA also used VisualGenome, and we wanted to use _all_ of them!

TBA.

## Exporting instances

TBA.
