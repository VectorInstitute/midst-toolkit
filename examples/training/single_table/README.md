# Single-Table Training Example

This example will go over training a single-table diffusion model from the ground up using the
code in this toolkit.


## Downloading data

First, we need the data. Download it from this
[Google Drive link](https://drive.google.com/file/d/1J5qDuMHHg4dm9c3ISmb41tcTHSu1SVUC/view?usp=drive_link),
extract the files and place them in a `/data` folder in within this folder
(`examples/training/single_table`).

> [!NOTE]
> If you wish to change the data folder, you can do so by editing the `base_data_dir` attribute
> of the [`config.yaml`](config.yaml) file.

Here is a description of the files that have been extracted:
- `trans.csv`: The training data. It consists of information about bank transactions and it
contains 20,000 data points.
- `trans_domain.json`: Metadata about the columns in `trans.csv`, such as data types and sizes.
- `dataset_meta.json`: Metadata about the relationship between the tables. Since this is a
single-table example, it will only contain information about the `trans` table.


## Kicking off training

The [`config.yaml`](config.yaml) file contains the parameters for the training. Please take a
look at them before kicking off the training and edit them as necessary.

To kick off training, simply run the command below from the project's root folder:

```bash
python -m examples.training.single_table.run_training
```

It will save the result files inside a `/results` folder within this folder
(`examples/training/single_table`).

> [!NOTE]
> If you wish to change the save folder, you can do so by editing the `results_dir` attribute
> of the [`config.yaml`](config.yaml) file.

In the `/results/models/` folder, there will be a file called `None_trans_ckpt.pkl`,
which is a pickle file containing the training results. You can load it using Python's
`pickle` and it will yield an instance of
`midst_toolkit.models.clavaddpm.train.ModelArtifacts`, which contains the trained
diffusion model along with some additional metadata about the training process:

```python
import pickle
from midst_toolkit.models.clavaddpm.train import ModelArtifacts

results_file = Path("examples/training/single_table/results/models/None_trans_ckpt.pkl")

 with open(results_file, "rb") as f:
    result = pickle.load(f)

assert isinstance(result, ModelArtifacts)
```
