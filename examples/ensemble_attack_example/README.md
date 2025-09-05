# Ensemble Attack

## Data Processing
As the first step of the attack, we need to collect and split the data. The input data collected from all the attacks provided by the MIDST Challenge should be stored in `data_paths.data_paths` as defined by `config.yaml`. You can download and unzip the resources from [this Google Drive link](https://drive.google.com/drive/folders/1rmJ_E6IzG25eCL3foYAb2jVmAstXktJ1?usp=drive_link). Note that you can safely remove the provided shadow models with the competition resources since they are not used in this attack.

Make sure directories and JSON files specified in `data_paths` and  `data_processing_config` configurations in `examples/ensemble_attack_example/config.yaml` exist.

To run the whole data processing pipeline, run `run_attack.py` and set `pipeline.run_data_processing` to `true` in `config.yaml`. It reads data from `data_paths.midst_data_path` specified in config, and populates `data_paths.population_data` and `data_paths.processed_attack_data_path` directories.

Data processing steps for the MIDST challenge provided resources according to Ensemble attack are as follows:

Step 1: Collect all the **train data** from all the attack types (every train folder provided by the challenge) including black-box and white-box attack. `population_all.csv` will include a total of `867494` data points. To alter the attack types for train data collection, change `data_processing_config.data_processing_config` in `config.yaml`.

Step 2: Collect all the **challenge data points** from `train`, `dev` and `final` folders of `tabddpm_black_box`.  `challenge_points_all.csv` will include a total of `13896` data points.

Step 3: Save population data without and with challenge points. `population_all_no_challenge.csv` will include a total of `855644` data points, and `population_all_with_challenge.csv` will include a total of `869540` data points.

`population_all_with_challenge.csv` is used to create real train and test data. Note that a random subset of `40k` data points are sampled from `population_all_with_challenge.csv` and used as population (or real data). You can change the number of random samples by changing `data_processing_config.population_sample_size` in `config.yaml`.

To run the steps first make sure to activate your virtual environment and adjust `config.yaml`. Then run:


```python

python -m examples.ensemble_attack_example.run_attack

```

Or you can directly run the bash script:

```bash
 ./examples/ensemble_attack_example/run.sh

```


## Terminology
To be added....
