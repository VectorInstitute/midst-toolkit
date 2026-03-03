# CTGAN Ensemble Attack Example

On this example, we demonstrate how to run the [Ensemble Attack](examples/ensemble_attack)
using the [CTGAN](https://arxiv.org/pdf/1907.00503) model.

## Downloading data

First, we need the data. Download it from this
[Google Drive link](https://drive.google.com/file/d/1B9z4vh51mH6ZMj5E0pJitqR8lid3EJKM/view?usp=drive_link),
extract the files and place them in a `/data/ensemble_attack` folder in within this folder
(`examples/gan`).

> [!NOTE]
> If you wish to change the data folder, you can do so by editing the `base_data_dir` attribute
> of the [`config.yaml`](config.yaml) file.

Here is a description of the files that have been extracted:
- `master_challenge_train.csv`:
- `population_all_with_challenge.csv`:
- `dataset_meta.json`: Metadata about the relationship between the tables in the dataset. Since this is a
single table dataset, it will only contain information about the transaction (`trans`) table.
- `trans_domain.json`: Metadata about the columns of the transaction table, such as their size
and type (`continuous` or `discrete`).
- `data_types.json`: Additional metadata about the columns, splitting them into 4 types:
    - `numerical`: a list of the columns that contain numerical information
    - `categorical`: a list of the columns that contain categorical information
    - `variable_to_predict`: the name of the target column that will be predicted
    - `id_column_name`: the name of the column in the table that represents the rows' id.

With the data present in the correct folder, we can proceed with running the attack.

## Running the attack

> [!NOTE]
> In the [`config.yaml`](config.yaml) file, the attribute `ensemble_attack.shadow_trainig.model_name`
> is what determines this attack will be run with the CTGAN model.

To run the attack, execute the following command from the project's root folder:

```bash
python -m examples.gan.ensemble_attack.run
```

This will take a long time to run, so it might be a good idea to execute it as a
background process. If you want to have a quick test run before kicking off the
full process, you can change the number of iterations, epochs, population and
sample sizes to smaller numbers.

## Results
