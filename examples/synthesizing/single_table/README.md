# Single-Table Synthesizing Example

This example will go over synthesizing data for a single-table dataset from the ground
up using the code in this toolkit.


## Downloading data

First, we need the data. Download it from this
[Google Drive link](https://drive.google.com/file/d/1YbDRVn-fwfdcPnHj5eMhCa6A-YPiGnKr/view?usp=sharing),
extract the files and place them in a `/data` folder within this folder
(`examples/synthesizing/single_table`).

> [!NOTE]
> If you wish to change the data folder, you can do so by editing the `base_data_dir` attribute
> of the [`config.yaml`](config.yaml) file.

Here is a description of the files that have been extracted:
- `trans.csv`: The training data. It consists of information about bank transactions and it
contains 20,000 data points.
- `trans_domain.json`: Metadata about the columns in `trans.csv`, such as data types and sizes.
- `dataset_meta.json`: Metadata about the relationship between the tables. Since this is a
single-table example, it will only contain information about the `trans` table.


## Kicking off synthesizing

If there is a `/results` folder within this folder (`examples/synthesizing/single_table`)
from a previous training run, we will use that data to kick off synthesizing.
For example, you can copy the results from another run (e.g. `examples.training.single_table.run_training`)
and paste them here and it will be picked up by this example.

The [`config.yaml`](config.yaml) file contains the parameters for the synthesizing and also
for training, in case there is a need to run that. Please take a look at them before kicking
off the synthesizing process and edit them as necessary.

To kick off synthesizing, simply run the command below from the project's root folder:

```bash
python -m examples.synthesizing.single_table.run_synthesizing
```

## Results

It will save the result files inside a `/results` folder within this folder
(`examples/synthesizing/single_table`).

> [!NOTE]
> If you wish to change the save folder, you can do so by editing the `results_dir` attribute
> of the [`config.yaml`](config.yaml) file.

In the `/results/before_matching/` folder, there will be a file called `synthetic_tables.pkl`,
which is a pickle file containing the synthetic data before the matching process, in case
it's needed.

The `/results/single_table_synthesizing` folder will contain the final synthesized
data, organized per table. In this single-table example, there is only going to be one
synthesized table under `/results/single_table_synthesizing/trans/_final/trans_synthetic.csv`.
