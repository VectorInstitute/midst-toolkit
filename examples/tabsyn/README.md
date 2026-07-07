# TabSyn Single Table Example

This example will go over training a single-table [TabSyn](https://arxiv.org/abs/2310.09656)
model and synthesizing data afterwards.


## Downloading data

First, we need the data. Download it from this
[Google Drive link](https://drive.google.com/file/d/1HTgfgeL5GXc8uAGfeQirJrUynK7vFeyb/view?usp=drive_link),
extract the files and place them in a `/data` folder in within this folder
(`examples/tabsyn`).

> [!NOTE]
> If you wish to change the data folder, you can do so by editing the `base_data_dir` attribute
> of the [`config.yaml`](config.yaml) file.

Here is a description of the files that have been extracted:
- `trans.csv`: The training data. It consists of information about bank transactions and it
contains 20,000 data points.
- `trans_info.json`: Metadata about the `trans.csv` data, with information such as which columns are
numerical and which are categorical, what is the task type, etc.


## Kicking off training

To kick off training, simply run the command below from the project's root folder:

```bash
python -m examples.tabsyn.train
```


## Training results

The result files will be saved inside a `/results` folder within this folder
(`examples/tabsyn`).

> [!NOTE]
> If you wish to change the save folder, you can do so by editing the `results_dir` attribute
> of the [`config.yaml`](config.yaml) file.

In the `/results/trans` folder, there will be a file called `model.pt`,
which is a pytorch saved model.


## Synthesizing data

To synthesize some data with the trained model, run:

```bash
python -m examples.tabsyn.synthesize
```

If there is already a trained model in the `/results` folder, it will use that model.
Otherwise it will train one from scratch. At the end of the script, it will save the
synthesized data to `/results/trans/synthetic_data/trans_synthetic.csv`.


## Evaluating the quality of the synthetic data

### Alpha Precision

To run a round of evaluation with [Alpha Precision](https://arxiv.org/abs/2301.07573)
metrics on a set of synthetic data, run the `evaluate.py` script:

```bash
python -m midst_toolkit.evaluation.quality.scripts.midst_alpha_precision_eval \
  --synthetic_data_path examples/tabsyn/results/trans/synthetic_data/trans_synthetic.csv \
  --real_data examples/tabsyn/data/trans_sampled.csv \
  --meta_info_path examples/tabsyn/data/meta_info.json \
  --save_directory examples/tabsyn/results/
```

It will save the evaluation results under the `/results/model.txt` file.

### Additional Metrics

The calculation of additional metrics are set up in the `evaluate.py` file. They are the
Kolmogorov-Smirnov (KS) test, Total Variation Distance (TVD), Correlation Matrix Difference
and Mutual Information Difference.

To compute those metrics, you can run the command below. The data files should
be under `/data/{table_name}.csv` for the real data, `/data/{table_name}_sampled.csv`
for the sampled data used for training, and `/results/{table_name}_synthetic.csv`
for the synthetic data.

```bash
python -m examples.tabsyn.evaluate
```

The results will be saved in the `/results/evaluation.json` file.
