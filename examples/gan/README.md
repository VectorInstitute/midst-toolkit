# CTGAN Single-Table Example

This example will go over training a single-table [CTGAN](https://arxiv.org/pdf/1907.00503)
model using the [CTGAN](https://github.com/sdv-dev/CTGAN/) library and then synthesizing
some data afterwards.


## Downloading data

First, we need the data. Download it from this
[Google Drive link](https://drive.google.com/file/d/1J5qDuMHHg4dm9c3ISmb41tcTHSu1SVUC/view?usp=drive_link),
extract the files and place them in a `/data` folder in within this folder
(`examples/gan`).

> [!NOTE]
> If you wish to change the data folder, you can do so by editing the `base_data_dir` attribute
> of the [`config.yaml`](config.yaml) file.

Here is a description of the files that have been extracted:
- `trans.csv`: The training data. It consists of information about bank transactions and it
contains 20,000 data points.
- `trans_domain.json`: Metadata about the columns in `trans.csv`, such as data types and sizes.
- `dataset_meta.json`: Metadata about the relationship between the tables. Since this is a
single-table example, it will only contain information about the `trans` table.
- `meta_info.json`: Metadata about the dataset, namely which columns are numerical and
which ones are categorical, the target column and the task type (e.g. `regression`).


## Kicking off training

To kick off training, simply run the command below from the project's root folder:

```bash
python -m examples.gan.train
```


## Training results

The result files will be saved inside a `/results` folder within this folder
(`examples/gan`).

> [!NOTE]
> If you wish to change the save folder, you can do so by editing the `results_dir` attribute
> of the [`config.yaml`](config.yaml) file.

In the `/results` folder, there will be a file called `trained_ctgan_model.pkl`,
which is a pickle file containing the trained model. You can load it using CTGAN's
`load` function:

```python
import pickle
from ctgan import CTGAN

results_file = Path("examples/gan/results/trained_ctgan_model.pkl")

ctgan = CTGAN.load(results_file)
```

## Synthesizing data

To synthesize some data with the trained model, run:

```bash
python -m examples.gan.synthesize
```

If there is already a trained model in the `/results` folder, it will use that model.
Otherwise it will train one from scratch. At the end of the script, it will save the
synthesized data to `/results/trans_synthetic.csv`.


## Evaluating the quality of the synthetic data

### Alpha Precision

To run a round of evaluation with [Alpha Precision](https://arxiv.org/abs/2301.07573)
metrics on a set of synthetic data, run the `evaluate.py` script:

```bash
python -m midst_toolkit.evaluation.quality.scripts.midst_alpha_precision_eval \
  --synthetic_data_path examples/gan/results/trans_synthetic.csv \
  --real_data examples/gan/data/trans.csv \
  --meta_info_path examples/gan/data/meta_info.json \
  --save_directory examples/gan/results/
```

It will save the evaluation results under the `/results/model.txt` file.

### Additional Metrics

The calculation of additional metrics are set up in the `evaluate.py` file. They are the
Kolmogorov-Smirnov (KS) test, Total Variation Distance (TVD), Correlation Matrix Difference
and Mutual Information Difference.

To compute those metrics, you can run the command below. The name of the table should be
defined in the `dataset_meta.json` file, and the file for synthetic data should be under
`/data/{table_name}.csv` for the real data and `/results/{table_name}_synthetic.csv`
for the synthetic data.

```bash
python -m examples.gan.evaluate
```

The results will be saved in the `/results/evaluation.json` file.
