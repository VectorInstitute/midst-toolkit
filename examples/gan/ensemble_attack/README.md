# CTGAN Ensemble Attack Example

On this example, we demonstrate how to run the [Ensemble Attack](examples/ensemble_attack)
using the [CTGAN](https://arxiv.org/pdf/1907.00503) model.

## Downloading data

First, we need the data. Download it from this
[Google Drive link](https://drive.google.com/file/d/1B9z4vh51mH6ZMj5E0pJitqR8lid3EJKM/view?usp=sharing),
extract the files and place them in a `/data/ensemble_attack` folder in within this folder
(`examples/gan`).

> [!NOTE]
> If you wish to change the data folder, you can do so by editing the `base_data_dir` attribute
> of the [`config.yaml`](config.yaml) file.

Here is a description of the files that have been extracted:
- `master_challenge_train.csv`:
- `population_all_with_challenge.csv`:

## Running the attack

To run, execute the following command from the prooject's root folder:

```bash
python -m examples.gan.ensemble_attack.run
```
