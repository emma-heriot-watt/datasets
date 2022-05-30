# Preparing the raw metadata for the datasets

Importantly, we try to avoid downloading images and any media/features for the datasets. We only care about getting the metadata for a dataset, putting it all together nicely, and then pumping it out fast!

## Downloading the raw data

There is a single CSV file which contains a link to every file which needs to be downloaded to create the full dataset.

You can automatically download all the files by running:

```bash
python -m emma_datasets download
```

### Downloading specific datasets

If you want to download specific datasets, you can provide the dataset name to the command. For example, to download just COCO and GQA, run:

```bash
python -m emma_datasets download coco gqa
```

## Organising the raw data

A script has been created to automatically extract and organise the raw data for you. You can automatically do this by calling

```bash
python -m emma_datasets organise
```

### Organising specific datasets

You can automatically organise specific datasets by providing the dataset name, [similar to above](#downloading-specific-datasets).

## Updating the dataset sources

All the remote paths where raw data can be downloaded from are stored in `src/emma_datasets/constants/dataset_downloads.csv`.

The first column of dataset names **must** correspond to the enum names for `DatasetName`, found in `src/emma_datasets/datamodels/constants.py`
