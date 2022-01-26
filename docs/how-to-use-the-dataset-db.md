# DatasetDB: EMMA's dataset manager

We implemented `DatasetDB`, a dedicated database for storing datasets that can be easily
processed in PyTorch. It provides a simple interface to access, iterate and create datasets. It is
based on [SQLite](https://www.sqlite.org/index.html) so it avoids loading in memory all the dataset
content which is perfect for multi-process training.

If you're interested in
understanding how it was implemented, please go the `Dataset structure` section.

## How do I use it?

A `DatasetDB` can be instantiated using a Python context manager to make sure that the underlying
database connection is correctly closed. This can be done as follows:

```python
with DatasetDB("path/to/dataset.db") as db:
  # now you can use the db...
```

The connection to the database will be closed automatically when the object goes out of scope.

### Reading from a database

Each dataset is composed of several examples. Each example in this library is represented as a
tuple `(data_id, example_id, data)`:

- `data_id` is the instance index;
- `example_id` is the identifier used by the dataset to represent the current instance
- `data` is a byte representation of the instance's content.

By default, the instance content is assumed to be JSON, so the `DatasetDB` will return a Python
object when reading from the underlying SQLite database.

To access the data, you can iterate over them as follows:

```python
from src.api.storage import DatasetDB

with DatasetDB("path/to/dataset.db") as db:
  for data_id, example_id, data in db:
    # do something with the fields...
```

You can access to a specific instance using either type of identifier. The `DatasetDB` can be
used as a Python dictionary:

```python
from src.api.storage import DatasetDB

with DatasetDB("path/to/dataset.db") as db:
  # the `data_id` has to be of type `int`
  data_id = 150
  instance = db[data_id]

  # the `example_id` has to be of type `str`
  example_id = "pretraining_150"
  instance = db[example_id]
```

### Integration in Pytorch

The previous examples are useful if you are just interested in exploring the data. However, one
important use case for EMMA is to use the data to train a PyTorch model. We can use the `DatasetDB`
as follows:

```python
from src.api.storage import DatasetDB
from torch.utils.data import Dataset

class EmmaPretrainingDataset(Dataset):
  def __init__(self, db_path):
    self.db = DatasetDB(db_path)

  def __len__(self):
    # Don't worry, this is extremely efficient because we have an index on the primary key :)
    return len(self.db)

  def __getitem__(self, index):
    instance = self.db[index]

    # I'm assuming you have a way to transform your raw JSON data to tensors
    tensors = transform(instance)

    return tensors

```

### Writing to a database

We can create a `DatasetDB` using a similar API which is described in the following code snippet:

```python
from src.api.storage import DatasetDB

num_instances = 10

with DatasetDB("path/to/dataset.db", readonly=False) as db:
  for data_id in range(num_instances):
    # this is just an example, you can use any Python object
    instance = {"caption": "This is a caption"}
    example_id = f"instance_{data_id}"
    db[(data_id, example_id)] = instance
```

In this snippet, we assume that the dataset creation happens once so that we are able to assign one
unique `data_id` to each instance. In general, `data_id` represents an index that goes from `0` to
`N-1`, where `N` is the number of datapoints in the dataset.

When writing to the database, you may want to adjust the parameter `batch_size` (default=`512`).
This represents the number of instances in the database cache that are retained before we flush its
content into the database.

## How is it implemented?

Thanks to SQLite integration in Python, we could implement this in pure Python code. The actual
implementation can be found in the [storage module](../src/api/storage.py).

### Database structure

SQLite is a powerful and efficient relational database that we use for storing the dataset we are
interested in. We assume that a dataset is composed of `N` datapoints `[x_1, x_2, ..., x_N]`.
In order to represent it in a relational database, we define a table `dataset` that has the
following columns:

- `data_id`: an instance counter for all the database instances (defined as `INTEGER PRIMARY KEY`)
- `example_id`: an identifier for the instance (defined as `TEXT`)
- `data`: the instance's content in byte (defined as `BLOB`, see `Storage types` for details)

### Storage types

The underlying SQLite does not support specific Python objects out of the box. Therefore, we
serialise all the instance data in bytes and store them in a `BLOB` field. At the moment, we
support two different storage types:

1. `TorchStorage`: This storage uses the default PyTorch serialisation format and can be used to
   store any PyTorch/Python object. For more details refer to the [official documentation](https://pytorch.org/docs/stable/notes/serialization.html).
2. `JsonStorage`: We use the custom [Orjson](https://github.com/ijl/orjson) library that also supports NumPy serialisation.

By default, `JsonStorage` is used as serialisation type for all the instances. If you're interested
in storing actual PyTorch tensors, you can change the serialisation format as follows:

```python
from src.api.storage import DatasetDB, StorageType

db = DatasetDB("/path/to/dataset.db", storage_type=StorageType.torch)

```
