# How to add a new downstream dataset

A **downstream dataset** is one that will be used for evaluating the model after all the training, and not for pretraining. That means each DB will only have the instances **for a single dataset**.

For downstream dataset instances, we are making the assumption that we do not need to do any annotation extracting or anything, and that it can all be done together in one go.

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
1. Include your new dataset name in `DatasetModalityMap` (in `emma_datasets.datamodels.constants`)
1. Based on the annotations your dataset has, add it to `AnnotationDatasetMap` (in
   `emma_datasets.datamodels.constants`)

## Preparing to create tests for the new dataset

Datasets are huge and there might be edge cases you probably didn't consider. To reduce the likelihood of wasting time on manually checking the full pipeline for bugs, you can implement smaller fine-grained tests to verify and isolate any issues.

To prepare, you need to do the following:

1. Download a subset of instances and include the file(s) in `storage/fixtures/` — if you have multiple files, store them in a folder.
1. Create a fixture in `tests/fixtures/paths.py` which can point to the source data file/folder. Be sure to use `fixtures_root`.

That's it for now! We'll come back to using these files later in the guide.

## Creating models for a single dataset instance

We need to easily parse every single instance and validate them to ensure that everything in the raw data is as you expect it! These classes are what get exported and stored within the DB file, and can be imported straight from the DB within your model.

Using [Pydantic's parsing methods](https://pydantic-docs.helpmanual.io/usage/models/#helper-functions), your raw data will be loaded from this file directly, and it will be validated to the schema of this model. This way, we can also verify that every single instance of the model is what you expect it to be, and you can deal with any weird data issues if they get flagged. If this happens, it will error.

The best example for what you are capable of doing is `TeachEdhInstance` (in `emma_datasets.datamodels.datasets.teach`). Each instance is represented by a single `TeachEdhInstance` and is used to import. It's made of multiple models, and Pydantic automatically parses all the nested models for you!

We advocate that the model for your dataset instance is as detailed as possible, using additional validators and properties where possible to ease usage downstream and ensure _every single instance_ is as expected.

### Creating an instance for your dataset

1. Create a new file for your dataset in `src/emma_datasets/datamodels/datasets/`.
1. Import `BaseInstance` from `emma_datasets.datamodels.base_model`
1. Create a class for your instance, inheriting from `BaseInstance`. For consistency, please ensure that the name of your class is in pascal case.
1. Import your new instance into `emma_datasets/datamodels/datasets/__init__.py` to ensure it is easily accessible like the others.

_Note: We used Pascal case because we made some typos and we don't want to rename every single class throughout the library, so it's just staying that way for now._

#### What if the raw data is not in `snake_case`?

If the keys in the raw data are not in snake case, you can use the `alias` key in `Field`. Check out [the example here](https://pydantic-docs.helpmanual.io/usage/model_config/#alias-precedence) or in ALFRED-related models (in `src/emma_datasets/datamodels/datasets/alfred.py`)

### Testing your new instance

We want to verify that your data can be imported and parsed by your instance correctly. As shown by `tests/datamodels/test_teach_datamodels.py`, you can verify that your model is working on your raw data correctly without needing to run the entire pipeline.

Using your [previously created fixture paths](#preparing-to-create-tests-for-the-new-dataset), create a new module for your dataset within `tests/datamodels` and add your tests to it.

If you are unsure what tests to include, check out other tests within the same directory. If you include `@property`'s in your instance, then you'll also want to test them too as they should be present for all instances.

## Adding the new dataset to the `downstream` command

To keep things simple for user, all DBs for downstream datasets can be created using the `emma_datasets` CLI. Each dataset has its own subcommand, and adding a new one should be nice and simple. You can find all subcommands for the `downstream` command are in `src/emma_datasets/commands/create_downstream_dbs.py`.

### Creating the command for the CLI

1. Create a new function which will store the logic for your downstream dataset — you can use `pass` for now for the content
1. Decorate your function with `@app.command("name")`, where `name` is the name of your dataset and will be accessible by the commands. Ensure that the commands are using hyphens and not underscores.
1. Add 3 arguments to the function:
   1. One pointing to the source directory of instances
   1. `output_dir`: the output directory of the created DB files
   1. `num_workers`: the number of workers using in the multiprocessing pool
1. Add any other arguments/options to the function, these will also be available by the command.
1. Add a docstring for the function. This will also be the help for the subcommand within the CLI. You can also use everything [Typer]() has to offer to include more advanced functionality.

You can test your command exists within the CLI by running:

```bash
python -m emma_datasets downstream --help
```

Your new function/command should be listed as a possible command, and you should be able to run it.

### Adding the DB creator to your new command

You can use the `DownstreamDbCreator` to easily process all the files for all the dataset splits. There are two `@classmethod`s that you can use to simplify the entire process:

- `DownstreamDbCreator.from_one_instance_per_json()`: for when each instance is within a separate JSON file
- `DownstreamDbCreator.from_jsonl()`: for when all instances are contained within a single JSONL file

For each class method, the process is mostly identical.

1. Create a dictionary pointing to separate paths for each dataset split for your dataset
2. Import the model for your instance from `emma_datasets.datamodels.datasets` (since you added it to the `__init__` above)
3. Instantiate the class using the class method
4. Call `.run(num_workers)`, providing the `num_workers`

### Testing your new command

After everything, we want to ensure that we can create the DBs, and that the command is working as expected. Importantly, we want to do this separately because if there are errors, we will want to know whether the problem is within the `DownstreamDbCreator` or caused by the downstream command function.

#### Testing your instance with the `DownstreamDbCreator`

1. Go to `tests/test_downstream_db_creator.py`
2. Create a function to test the `DownstreamDbCreator` on your dataset. You _can_ just copy-paste other functions and tweak the aspects necessary for your dataset. This doesn't need to be elegant, it needs to robustly test that everything works as expected.

#### Testing your new subcommand works

1. Go to `tests/commands/test_create_downstream_dbs.py`
2. Create a function to test your command. You _can_ just copy-paste from other test functions and tweak the aspects necessary for your command. Again, this doesn't need to be elegant, it needs to robustly test everything works.
