"""Build a balanced challenge dataset from train and population CSVs.

Non-train rows are population rows whose ID is not in the training set. The script
samples an equal count from train and from non-train, concatenates them, and writes
challenge features and membership labels (1 = train, 0 = non-train) for MIA evaluation.

python examples/ensemble_attack/diabetes/create_challenge_data.py \
  --output-model-directory  /path/to/tabddpm_8 \
  --train-path /path/to/train_with_id.csv \
  --population-path /path/to/population.csv \
  --num-per-split 500 \
  --random-seed 42


python -m examples.ensemble_attack.diabetes.create_challenge_data \
  --output-model-directory  /projects/midst-experiments/ensemble_attack/diabetes_experiments/10k_default/1k_challange_points_dir/tabddpm_10 \
  --train-path /projects/midst-experiments/diabetes_experiments/whitebox_single_table_DI_1/tabddpm_10/train_with_id.csv \
  --population-path /projects/midst-experiments/ensemble_attack/diabetes_experiments/10k_default/all_train_with_id.csv \
  --num-per-split 500 \
  --random-seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create challenge_with_id.csv and challenge_label.csv under the model directory: "
            "sample N rows from train and N from population excluding train IDs."
        )
    )
    parser.add_argument(
        "--output-model-directory",
        type=Path,
        required=True,
        help="Directory where challenge_with_id.csv and challenge_label.csv are written.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        required=True,
        help="Path to the training CSV (e.g. train_with_id.csv).",
    )
    parser.add_argument(
        "--population-path",
        type=Path,
        required=True,
        help="Path to the population CSV (superset from which non-train rows are derived).",
    )
    parser.add_argument(
        "--num-per-split",
        type=int,
        required=True,
        help="Number of rows to sample from train and the same number from non-train "
        "(challenge size will be 2 * this value).",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="encounter_id",
        help="Column name used to match train rows when filtering population (default: encounter_id for diabetes).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for reproducible sampling.",
    )
    parser.add_argument(
        "--challenge-data-filename",
        type=str,
        default="challenge_with_id.csv",
        help="Output filename for challenge rows inside output-model-directory .",
    )
    parser.add_argument(
        "--challenge-label-filename",
        type=str,
        default="challenge_label.csv",
        help="Output filename for labels inside output-model-directory .",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_model_dir = args.output_model_directory.expanduser().resolve()
    output_model_dir.mkdir(parents=True, exist_ok=True)


    id_col = args.id_column
    n = args.num_per_split

    df_train = pd.read_csv(args.train_path)
    df_pop = pd.read_csv(args.population_path)

    for name, df in ("train", df_train), ("population", df_pop):
        if id_col not in df.columns:
            raise ValueError(f"{name} data is missing id column {id_col!r}. Columns: {list(df.columns)}")

    train_ids = df_train[id_col].unique()
    non_train_mask = ~df_pop[id_col].isin(set(train_ids))
    df_non_train = df_pop.loc[non_train_mask].copy()

    if len(df_train) < n:
        raise ValueError(
            f"Need at least {n} train rows, got {len(df_train)}. Reduce --num-per-split."
        )
    if len(df_non_train) < n:
        raise ValueError(
            f"Need at least {n} non-train rows after excluding train IDs, got {len(df_non_train)}. "
            "Check population coverage or reduce --num-per-split."
        )

    sample_train = df_train.sample(n=n, random_state=args.random_seed).copy()
    sample_non = df_non_train.sample(n=n, random_state=args.random_seed).copy()

    challenge_df = pd.concat([sample_train, sample_non], axis=0, ignore_index=True)
    labels = pd.DataFrame({"label": [1] * n + [0] * n})

    out_data = output_model_dir / args.challenge_data_filename
    out_label = output_model_dir / args.challenge_label_filename

    challenge_df.to_csv(out_data, index=False)
    labels.to_csv(out_label, index=False)

    print(f"Wrote {len(challenge_df)} rows to {out_data}")
    print(f"Wrote labels ({n} train=1, {n} non-train=0) to {out_label}")


    # Now remove these two files if they exist
    # challenge_label_predictions.csv  data_for_validating_MIA.csv
    if (output_model_dir / "challenge_label_predictions.csv").exists():
        (output_model_dir / "challenge_label_predictions.csv").unlink()
    if (output_model_dir / "data_for_validating_MIA.csv").exists():
        (output_model_dir / "data_for_validating_MIA.csv").unlink()
    if (output_model_dir / "data_for_training_MIA.csv").exists():
        (output_model_dir / "data_for_training_MIA.csv").unlink()


if __name__ == "__main__":
    main()
