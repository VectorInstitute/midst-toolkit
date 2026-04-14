"""Collect train_with_id.csv from each subfolder under a base directory and concatenate.

Example usage:
python -m examples.ensemble_attack.diabetes.collect_train_with_id --base-directory /projects/midst-experiments/diabetes_experiments/whitebox_single_table_DI_1/ --output-directory /projects/midst-experiments/ensemble_attack/diabetes_experiments/10k_default/

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find train_with_id.csv in each folder under --base-directory, concatenate rows, "
            "and write a single CSV to --output-directory."
        )
    )
    parser.add_argument(
        "--base-directory",
        type=Path,
        required=True,
        help="Root directory whose subfolders (or nested paths, see --recursive) contain train_with_id.csv.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory where the combined CSV is written (created if missing).",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="all_train_with_id.csv",
        help="Name of the output file inside --output-directory.",
    )
    parser.add_argument(
        "--recursive",
        default=True,
        type=bool,
        help=(
            "If set, discover every train_with_id.csv under base-directory (rglob). "
            "Default: only immediate child folders of base-directory."
        ),
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="train_with_id.csv",
        help="CSV filename to look for in each folder.",
    )
    return parser.parse_args()


def discover_csv_paths(base: Path, csv_name: str, recursive: bool) -> list[Path]:
    if not base.is_dir():
        raise FileNotFoundError(f"Base directory does not exist or is not a directory: {base}")

    paths: list[Path] = []
    if recursive:
        paths = sorted(base.rglob(csv_name))
    else:
        for child in sorted(base.iterdir()):
            if child.is_dir():
                candidate = child / csv_name
                if candidate.is_file():
                    paths.append(candidate)

    return paths


def main() -> None:
    args = parse_args()
    base = args.base_directory.expanduser().resolve()
    out_dir = args.output_directory.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_csv_paths(base, args.csv_name, args.recursive)
    if not paths:
        loc = "subfolders" if not args.recursive else "tree"
        raise FileNotFoundError(
            f"No {args.csv_name!r} found under {base} ({loc}). "
            f"Try --recursive if files live in nested directories (e.g. final/tabddpm_8/)."
        )

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)

    combined = pd.concat(frames, axis=0, ignore_index=True)
    out_path = out_dir / args.output_filename
    combined.to_csv(out_path, index=False)

    print(f"Merged {len(paths)} file(s) -> {len(combined)} rows")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
