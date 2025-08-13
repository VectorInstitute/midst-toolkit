import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from midst_toolkit.evaluation.generation.quality import synthcity_alpha_precision_metrics
from midst_toolkit.evaluation.generation.utils import (
    create_quality_metrics_directory,
    dump_metrics_dict,
    extract_columns_based_on_meta_info,
    one_hot_encode_categoricals_and_merge_with_numerical,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Killing a benign pandas warning
pd.options.mode.chained_assignment = None


def load_midst_data(
    real_data_path: Path, synthetic_data_path: Path, meta_info_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Helper function for loading data at the specified paths. These paths are constructed either by the user or with a
    particular set of defaults that were used in the original MIDST competition (see, for example,
    https://github.com/VectorInstitute/MIDSTModels/blob/main/midst_models/single_table_TabDDPM/eval/eval_quality.py).

    Args:
        real_data_path: Path from which to load the real data to which the synthetic data will be compared. This
            should be a CSV file.
        synthetic_data_path: Path from which to load the synthetic data to which the real data will be compared. This
            should be a CSV file.
        meta_info_path: This should be a JSON file containing meta information about the data generation process.
            Specifically, it should contain information about which columns of the real and synthetic data should
            actually be compared. It must contain keys: 'num_col_idx', 'cat_col_idx', 'target_col_idx', and
            'task_type'.

    Returns:
        The loaded real data, synthetic data, and meta information json for further processing.
    """
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    with open(meta_info_path, "r") as f:
        meta_info = json.load(f)

    return real_data, synthetic_data, meta_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic_data_path",
        required=False,
        type=str,
        default=None,
        help="The file path of the synthetic data. If specified, overrides dataname and model args",
    )
    parser.add_argument(
        "--real_data_path",
        required=False,
        type=str,
        default=None,
        help="The file path of the real data. If specified, overrides dataname and model args",
    )
    parser.add_argument(
        "--meta_info_path",
        required=False,
        type=str,
        default=None,
        help="The file path of the meta data for the synthetic generation. If specified, overrides dataname",
    )
    parser.add_argument(
        "--dataname",
        required=True,
        type=str,
        default="adult",
        help=(
            "Used to construct default paths for the real or synthetic data if not specified. "
            "Also used to potentially trigger dataset preprocessing if one of 'default', 'shoppers', 'faults', 'news'"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        default="model",
        help=(
            "Used to construct default paths for the real or synthetic data if not specified. "
            "Also used to potentially trigger dataset preprocessing if '...codi', '...great' specified. "
        ),
    )

    args = parser.parse_args()

    dataname = args.dataname
    model = args.model
    synthetic_data_path = args.synthetic_data_path
    real_data_path = args.real_data_path

    if args.synthetic_data_path:
        synthetic_data_path = Path(args.synthetic_data_path)
    else:
        assert dataname is not None and model is not None, (
            "Both dataname and model must be defined to construct default synthetic data path"
        )
        synthetic_data_path = Path(f"synthetic/{dataname}/{model}.csv")
        LOGGER.info(f"Synthetic data path not provided. Default constructed: {synthetic_data_path}")

    if args.real_data_path:
        real_data_path = Path(args.real_data_path)
    else:
        assert dataname is not None, "Dataname must be defined to construct default real data path"
        real_data_path = Path(f"synthetic/{dataname}/real.csv")
        LOGGER.info(f"Real data path not provided. Default constructed: {real_data_path}")

    if args.meta_info_path:
        meta_info_path = Path(args.meta_info_path)
    else:
        assert dataname is not None, "Dataname must be defined to construct default meta info path"
        meta_info_path = Path(f"data/{dataname}/info.json")
        LOGGER.info(f"Meta info path not provided. Default constructed: {meta_info_path}")

    real_data, synthetic_data, meta_info = load_midst_data(real_data_path, synthetic_data_path, meta_info_path)

    numerical_real_data, categorical_real_data = extract_columns_based_on_meta_info(real_data, meta_info)
    numerical_synthetic_data, categorical_synthetic_data = extract_columns_based_on_meta_info(
        synthetic_data, meta_info
    )

    numerical_real_numpy = numerical_real_data.to_numpy()
    categorical_real_numpy = categorical_real_data.to_numpy().astype("str")

    numerical_synthetic_numpy = numerical_synthetic_data.to_numpy()
    categorical_synthetic_numpy = categorical_synthetic_data.to_numpy().astype("str")

    # Perform some special data post-processing for specific datasets and models as specified in the script
    # arguments

    if dataname in ["default", "news"] and model[:4] == "codi":
        # If using the default or news dataset and a model postfixed with "codi," need to perform an int cast
        categorical_synthetic_numpy = categorical_synthetic_data.astype("int").to_numpy().astype("str")
    elif model[:5] == "great":
        if dataname == "shoppers":
            # Column reassignment
            categorical_synthetic_numpy[:, 1] = categorical_synthetic_data[11].astype("int").to_numpy().astype("str")
            categorical_synthetic_numpy[:, 2] = categorical_synthetic_data[12].astype("int").to_numpy().astype("str")
            categorical_synthetic_numpy[:, 3] = categorical_synthetic_data[13].astype("int").to_numpy().astype("str")

            # Clip the maximum value to reflect that of the real data
            max_data = categorical_real_data[14].max()
            categorical_synthetic_data.loc[categorical_synthetic_data[14] > max_data, 14] = max_data

            # Perform column reassignment
            categorical_synthetic_numpy[:, 4] = categorical_synthetic_data[14].astype("int").to_numpy().astype("str")
            categorical_synthetic_numpy[:, 4] = categorical_synthetic_data[14].astype("int").to_numpy().astype("str")

        elif dataname in ["default", "faults", "beijing"]:
            # Note that columns here are not contiguous, so we enumerate
            columns = categorical_real_data.columns
            for i, col in enumerate(columns):
                if categorical_real_data[col].dtype == "int":
                    max_data = categorical_real_data[col].max()
                    min_data = categorical_real_data[col].min()

                    # Perform clipping based on the real data on both sides (min and max)
                    categorical_synthetic_data.loc[categorical_synthetic_data[col] > max_data, col] = max_data
                    categorical_synthetic_data.loc[categorical_synthetic_data[col] < min_data, col] = min_data

                    categorical_synthetic_numpy[:, i] = (
                        categorical_synthetic_data[col].astype("int").to_numpy().astype("str")
                    )

    real_dataframe, synthetic_dataframe = one_hot_encode_categoricals_and_merge_with_numerical(
        categorical_real_numpy, categorical_synthetic_numpy, numerical_real_numpy, numerical_synthetic_numpy
    )

    LOGGER.info("=========== All Features ===========")
    LOGGER.info(f"Data shape: {synthetic_dataframe.shape}")

    quality_results = synthcity_alpha_precision_metrics(real_dataframe, synthetic_dataframe, naive_only=True)
    save_dir = Path(f"eval/quality/{dataname}")
    create_quality_metrics_directory(save_dir)
    dump_metrics_dict(quality_results, save_dir / f"{model}.txt")
