# Multi-Table Synthesizing Example

This example will go over synthesizing data for a multi-table dataset from the ground
up using the code in this toolkit.


## Downloading data

First, we need the data. Download it from this
[Google Drive link](https://drive.google.com/file/d/1Ao222l4AJjG54-HDEGCWkIfzRbl9_IKa/view?usp=drive_link),
extract the files and place them in a `/data` folder in within this folder
(`examples/synthesizing/multi_table`).

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


## Kicking off synthesizing

If there is a `/results` folder within this folder (`examples/synthesizing/multi_table`)
from a previous training run, we will use that data to kick off synthesizing.
For example, you can copy the results from another run (e.g. `examples.training.multi_table.run_training`)
and paste them here and it will be picked up by this example.

The [`config.yaml`](config.yaml) file contains the parameters for the synthesizing and also
for training, in case there is a need to run that. Please take a look at them before kicking
off the synthesizing process and edit them as necessary.

To kick off synthesizing, simply run the command below from the project's root folder:

```bash
python -m examples.synthesizing.multi_table.run_synthesizing
```

## Results

It will save the result files inside a `/results` folder within this folder
(`examples/synthesizing/multi_table`).

> [!NOTE]
> If you wish to change the save folder, you can do so by editing the `results_dir` attribute
> of the (`config.yaml`)[config.yaml] file.

In the `/results/before_matching/` folder, there will be a file called `synthetic_tables.pkl`,
which is a pickle file containing the synthetic data before the matching process, in case
it's needed.

The `/results/multi_table_synthesizing` folder will contain the final synthesized
data, organized per table, in the form of `.csv` files with the following naming pattern:
`/results/multi_table_synthesizing/{table_name}/_final/{table_name}_synthetic.csv`.
