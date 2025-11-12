# Multi-Table Training Example

This example will go over training a multi-table diffusion model from the ground up using the
code in this toolkit.


## Downloading data

First, we need the data. Download it from this
[Google Drive link](https://drive.google.com/file/d/1Ao222l4AJjG54-HDEGCWkIfzRbl9_IKa/view?usp=drive_link),
extract the files and place them in a `/data` folder in within this folder
(`examples/training/multi_table`).

> [!NOTE]
> If you wish to change the data folder, you can do so by editing the `base_data_dir` attribute
> of the (`config.yaml`)[config.yaml] file.

It will contain data for 8 tables: `account`, `card`, `client`, `disp`, `district`, `loan`, `order`,
and `trans`. For each table there will be two files:
- `{table_name}.csv`: The table's data.
- `{table_name}_domain.json`: Metadata about the columns in the table's data, such as data types and sizes.

Additionally, you will find one more file:
- `dataset_meta.json`: Metadata about the relationship between the tables. It will describe which tables
are associated with which other tables.


## Kicking off training

The [`config.yaml`](config.yaml) file contains the parameters for the training. Please take a
look at them before kicking off the training and edit them as necessary.

To kick off training, simply run the command below from the project's root folder:

```bash
python -m examples.training.multi_table.run_training
```

It will save the result files inside a `/results` folder within this folder
(`examples/training/multi_table`).

> [!NOTE]
> If you wish to change the save folder, you can do so by editing the `results_dir` attribute
> of the (`config.yaml`)[config.yaml] file.

One of the results file is `/results/cluster_ckpt.pkl`, which will contain the results
of the clustering step.

The other result files are in the `/results/models/` folder. They will be named after the
table relations defined in `dataset_meta.json`. For example: for the `("client", "account")`
relation, there will be a file called `client_account_ckpt.pkl`, which is a pickle file
containing the training results. You can load it using Python's `pickle` and it will yield
an instance of `midst_toolkit.models.clavaddpm.train.ModelArtifacts`, which contains the
trained diffusion model along with some additional metadata about the training process:

```python
import pickle
from midst_toolkit.models.clavaddpm.train import ModelArtifacts

results_file = Path("examples/training/multi_table/results/models/client_account_ckpt.pkl")

 with open(results_file, "rb") as f:
    result = pickle.load(f)

assert isinstance(result, ModelArtifacts)
```
