# Tartan Federer Attack

This example runs a Tartan–Federer membership inference attack using trained TabDDPM models. The pipeline optionally performs a data processing step to prepare population datasets and then executes the attack using a binary classifier.

## Data Processing

The data processing step constructs population datasets representing the real data available to the attacker.

A selected subset of `train_with_id.csv` files is collected from `tabddpm_1` to `tabddpm_6` located under:

```
examples/tartan_federer_attack/tabddpm_trained_with_20k/tabddpm_white_box
```

For each selected model, both `train_with_id.csv` and `challenge_with_id.csv` are loaded. All training datasets are merged into a single dataframe and all challenge datasets are merged into a single dataframe. Any training samples that also appear in the challenge dataset are removed, and duplicate samples are dropped based on configured identifier columns.

The model indices used to build the population datasets for training and validation are specified in the configuration file:

```yaml
data_processing_config:
  population_attack_indices_to_collect_for_training: [1, 2]
  population_attack_indices_to_collect_for_validation: [3, 4]
  model_type: tabddpm
  columns_for_deduplication: ['trans_id', 'balance']
```

## Running the Attack

Before running the attack, activate your virtual environment and update `config.yaml` as needed. From the top-level directory of the library, run:

```bash
python -m examples.tartan_federer_attack.run_attack
```
